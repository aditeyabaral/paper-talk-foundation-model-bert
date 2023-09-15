import os
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import DataCollatorForLanguageModeling

from chatbert.dataset import ChatBERTDataset
from chatbert.model import ChatBERT


class ChatBERTTrainer:
    def __init__(self, model, learning_rate, **kwargs):
        self.optimizer = None
        self.scheduler = None
        self.optimizer_class = kwargs.get("optimizer_class", "adamw")
        self.scheduler_class = kwargs.get("scheduler_class", "plateau")
        self.tensorboard_writer = None

        self.init_optimizer(model, learning_rate, **kwargs)

    def init_optimizer(self, model, learning_rate, **kwargs):
        optimizer_class = kwargs.get("optimizer_class", "adamw")
        if optimizer_class == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'adagrad':
            self.optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'sparse_adam':
            self.optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

        self.init_scheduler(**kwargs)

    def init_scheduler(self, **kwargs):
        scheduler_class = kwargs.get("scheduler_class", "plateau")
        if scheduler_class == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=1,
                verbose=True,
            )
        elif scheduler_class == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get("step_size", 1),
                gamma=kwargs.get("gamma", 0.1),
            )
        elif scheduler_class == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get("T_max", 10),
                eta_min=kwargs.get("eta_min", 0.0001),
            )
        elif scheduler_class == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=kwargs.get("gamma", 0.1),
            )
        else:
            raise NotImplementedError

    def init_tensorboard(self, tensorboard_log_dir):
        if Path(tensorboard_log_dir).exists():
            for file in Path(tensorboard_log_dir).iterdir():
                file.unlink()
        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir)

    def write_losses_to_tensorboard(self, losses, step):
        for key, value in losses.items():
            if key.startswith("loss") and value is not None and isinstance(value, torch.Tensor):
                self.tensorboard_writer.add_scalar(f"Loss/{key}", value, step)

    @staticmethod
    def freeze_components(model, freeze_components):
        for component in freeze_components:
            print(f"Freezing {component}...")
            if component == "mlm":
                model.freeze()
            elif component == "topic":
                model.topic_selector.freeze()
            else:
                print(f"Ignoring unknown component for freezing: {component}")

    @staticmethod
    def unfreeze_components(model, unfreeze_components):
        for component in unfreeze_components:
            print(f"Unfreezing {component}...")
            if component == "mlm":
                model.unfreeze()
            elif component == "topic":
                model.topic_selector.unfreeze()
            else:
                print(f"Ignoring unknown component for unfreezing: {component}")

    @staticmethod
    def compute_loss(outputs, **kwargs):
        loss = (
                kwargs.get("alpha", 1.0) * outputs.__dict__.get("loss_mlm", 0.0) +
                kwargs.get("beta", 1.0) * outputs.__dict__.get("loss_topic_association", 0.0) +
                kwargs.get("gamma", 1.0) * outputs.__dict__.get("loss_membership", 0.0) +
                kwargs.get("delta", 1.0) * outputs.__dict__.get("loss_cs", 0.0) +
                kwargs.get("eta", 1.0) * outputs.__dict__.get("loss_sender_cls", 0.0) +
                kwargs.get("zeta", 1.0) * outputs.__dict__.get("loss_recipient_cls", 0.0) +
                kwargs.get("kappa", 1.0) * outputs.__dict__.get("loss_nsp", 0.0)
        )
        return loss

    @staticmethod
    def get_masked_data_batch(masked_data, index, batch_size, device="cpu"):
        batch = dict()
        for key in masked_data:
            batch[key] = (masked_data[key][index: index + batch_size]).to(device)
        return batch

    def train(
            self,
            dataset: ChatBERTDataset,
            model: ChatBERT,
            batch_size=6,
            epochs=20,
            freeze_components=(),
            train_components=["mlm", "topic", "membership"],
            save_dir="./saved_models",
            device="cuda",
            **kwargs
    ):
        if epochs <= 0:
            return

        if kwargs.get("use_tensorboard", True):
            self.init_tensorboard(kwargs.get("tensorboard_log_dir", "./logs"))

        if kwargs.get("learning_rate", None) is not None:
            learning_rate = kwargs.get("learning_rate")
            self.optimizer.param_groups[0]["lr"] = learning_rate
            self.init_scheduler(**kwargs)

        if kwargs.get("upload_model_to_hub", False):
            self.clone_hub_repository_into_save_dir(save_dir, **kwargs)

        print(f"Training for {epochs} epochs "
              f"with components: {train_components} and frozen components: {freeze_components}")
        print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
        self.freeze_components(model, freeze_components)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=model.mlm_tokenizer,
            mlm_probability=kwargs.get("mlm_mask_probability", 0.15),
        )

        model.train()
        model.zero_grad()
        torch.cuda.empty_cache()
        total_conversational_examples = len(dataset.conversational_data)
        total_membership_examples = len(dataset.membership_data)
        # TODO: Rename these step variables
        mlm_topic_steps = 0
        membership_steps = 0
        topic_association_steps = 0
        for epoch in tqdm(range(1, epochs + 1)):
            num_mlm_topic_batches = 0
            total_epoch_loss = 0
            masked_data = data_collator(dataset.conversational_training_examples)

            for batch_idx in tqdm(range(0, total_conversational_examples, batch_size)):
                self.optimizer.zero_grad()

                masked_data_batch = None
                sender_cls_batch = None
                recipient_cls_batch = None
                next_sentence_prediction_batch = None

                if "mlm" in train_components:
                    masked_data_batch = self.get_masked_data_batch(
                        masked_data,
                        batch_idx,
                        batch_size,
                        device
                    )

                conversational_data_batch = dataset.conversational_data[batch_idx: batch_idx + batch_size]
                if "sender_cls" in train_components:
                    sender_cls_batch = conversational_data_batch
                if "recipient_cls" in train_components:
                    recipient_cls_batch = conversational_data_batch
                if "next_sentence_prediction" in train_components:
                    # TODO: All data transformations should be moved to dataset.py
                    next_sentence_prediction_batch = model.get_next_sentence_prediction_inputs_labels(
                        conversational_data_batch)

                outputs = model(
                    mlm_batch=masked_data_batch,
                    sender_cls_batch=sender_cls_batch,
                    recipient_cls_batch=recipient_cls_batch,
                )

                if next_sentence_prediction_batch:
                    # iterate through sub-batches because some messages are too long (> 20 sentences)
                    outputs.loss_nsp = 0
                    # TODO: Make this a parameter
                    nsp_batch_size = 4
                    for nsp_batch_index in range(0, len(next_sentence_prediction_batch[0]), nsp_batch_size):
                        nsp_batch = (
                            next_sentence_prediction_batch[0][nsp_batch_index: nsp_batch_index + nsp_batch_size],
                            next_sentence_prediction_batch[1][nsp_batch_index: nsp_batch_index + nsp_batch_size],
                            next_sentence_prediction_batch[2][nsp_batch_index: nsp_batch_index + nsp_batch_size],
                        )
                        nsp_outputs = model(next_sentence_prediction_batch=nsp_batch)
                        outputs.loss_nsp += nsp_outputs.loss_nsp.detach().cpu()
                        nsp_outputs.loss_nsp.backward(retain_graph=True)

                if self.tensorboard_writer is not None:
                    self.write_losses_to_tensorboard(outputs.__dict__, mlm_topic_steps)

                loss = self.compute_loss(outputs, **kwargs)

                if self.tensorboard_writer is not None:
                    self.write_losses_to_tensorboard({"loss": loss}, mlm_topic_steps)

                if isinstance(loss, torch.Tensor):
                    loss.backward()
                    total_epoch_loss += loss.detach().cpu().item()
                else:
                    total_epoch_loss += loss

                self.optimizer.step()
                mlm_topic_steps += 1
                num_mlm_topic_batches += 1

            total_epoch_loss = total_epoch_loss / num_mlm_topic_batches

            if "membership" in train_components:
                total_membership_loss = 0
                membership_batches = 0
                for batch_idx in tqdm(range(0, total_membership_examples, 1)):
                    self.optimizer.zero_grad()
                    membership_data_batch = list(dataset.membership_data.values())[batch_idx:batch_idx + batch_size]

                    outputs = model(membership_batch=membership_data_batch)

                    if self.tensorboard_writer is not None:
                        self.write_losses_to_tensorboard(outputs.__dict__, membership_steps)

                    if hasattr(outputs, "loss_membership"):
                        loss = outputs.loss_membership
                        total_membership_loss += loss.detach().cpu().item() \
                            if isinstance(loss, torch.Tensor) else loss
                        loss.backward()
                        self.optimizer.step()
                    membership_steps += 1
                    membership_batches += 1

                total_epoch_loss += total_membership_loss / membership_batches

            if "topic" in train_components:
                total_topic_association_loss = 0
                topic_association_batches = 0
                for batch_idx in tqdm(range(0, total_conversational_examples, 8)):
                    self.optimizer.zero_grad()
                    topic_association_batch = dataset.conversational_data[batch_idx:batch_idx + batch_size]

                    outputs = model(topic_association_batch=topic_association_batch)

                    if self.tensorboard_writer is not None:
                        self.write_losses_to_tensorboard(outputs.__dict__, topic_association_steps)

                    if hasattr(outputs, "loss_topic_association"):
                        loss = outputs.loss_topic_association
                        total_topic_association_loss += loss.detach().cpu().item() \
                            if isinstance(loss, torch.Tensor) else loss
                        loss.backward()
                        self.optimizer.step()
                    topic_association_steps += 1
                    topic_association_batches += 1

                total_epoch_loss += total_topic_association_loss / topic_association_batches

            self.scheduler.step(total_epoch_loss)
            print(f"Epoch {epoch} completed. Training Loss: {total_epoch_loss}. "
                  f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")

            self.push_training_state_to_hub(model, epoch, **kwargs)
            if epoch % kwargs.get("save_every", 1) == 0:
                self.save_training_state_locally(save_dir, model, epoch, **kwargs)
                if kwargs.get("upload_model_to_hub", False):
                    self.upload_model_to_hub(save_dir, epoch, **kwargs)

        self.push_training_state_to_hub(model, epochs + 1, **kwargs)
        self.save_training_state_locally(save_dir, model, epochs + 1, **kwargs)
        if kwargs.get("upload_model_to_hub", False):
            self.upload_model_to_hub(save_dir, epochs + 1, **kwargs)

        self.unfreeze_components(model, freeze_components)
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def save(self, path):
        if not Path(path).parent.exists():
            raise FileNotFoundError
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_class": self.optimizer_class,
                "scheduler_class": self.scheduler_class,
            }, path
        )

    def load(self, path):
        if not Path(path).exists():
            raise FileNotFoundError
        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer_class = checkpoint["optimizer_class"]
        self.scheduler_class = checkpoint["scheduler_class"]

    def save_training_state_locally(self, path, model: ChatBERT, epoch, **kwargs):
        if not Path(path).exists():
            Path(path).mkdir(parents=True)

        save_model_name = kwargs.get("save_model_name", "model.pt")
        if not kwargs.get("save_latest", False):
            base_model_name = Path(kwargs.get("save_model_name", "model.pt")).stem
            save_model_name = f"{base_model_name}_{epoch}.pt"
        model_save_path = Path(path).joinpath(save_model_name)
        trainer_save_path = Path(path).joinpath("trainer.pt")

        model.save(model_save_path)
        self.save(trainer_save_path)

    @staticmethod
    def push_training_state_to_hub(model: ChatBERT, epoch, **kwargs):
        if kwargs.get("auth_token", None) is None or kwargs.get("hub_model_name", None) is None:
            return

        repo_name = kwargs["hub_model_name"]
        if kwargs.get("hub_organization", None):
            repo_name = f"{kwargs['hub_organization']}/{kwargs['hub_model_name']}"

        model.push_to_hub(
            repo_name=f"{repo_name}-embeddings",
            commit_message=f"Epoch {epoch}",
            auth_token=kwargs.get("auth_token")
        )

    @staticmethod
    def upload_model_to_hub(path, epoch, **kwargs):
        if not Path(path).exists() or kwargs.get("auth_token", None) is None:
            return
        path = str(Path(path).absolute().resolve())
        command = f"cd \"{path}\" && git lfs install && huggingface-cli lfs-enable-largefiles . && git-lfs pull && " \
                  f"git gc && git add . && git commit -m \"Model: Epoch {epoch}\" && git push"
        os.system(command)

    @staticmethod
    def clone_hub_repository_into_save_dir(save_dir, **kwargs):
        if kwargs.get("auth_token", None) is None or kwargs.get("hub_model_name", None) is None:
            return
        auth_token = kwargs.get("auth_token")
        hub_model_name = kwargs.get("hub_model_name")
        hub_username = kwargs.get("hub_username", None)
        if hub_username is None:
            hub_username = os.popen("huggingface-cli whoami").read().strip().split('\n')[0]
            print(f"Using username {hub_username} for hub model {hub_model_name}")
        hub_organization = kwargs.get("hub_organization", "")
        hub_organization_flag = ""
        if hub_organization:
            hub_organization_flag = f"--organization {hub_organization}"

        print(f"Creating repository {hub_model_name}")
        p = subprocess.run(
            [
                "huggingface-cli",
                "repo",
                "create",
                hub_model_name,
                *hub_organization_flag.split(),
                "-y",
            ],
            capture_output=True,
        )
        print(p.stdout.decode("utf-8"))

        print(f"Cloning repository {hub_model_name} to {save_dir}")
        if hub_organization:
            clone_url = f"https://{hub_organization}:{auth_token}@huggingface.co/{hub_organization}/{hub_model_name}"
        else:
            clone_url = f"https://{hub_username}:{auth_token}@huggingface.co/{hub_username}/{hub_model_name}"

        p = subprocess.run(
            [
                "git",
                "clone",
                clone_url,
                save_dir
            ],
            capture_output=True
        )
        print(p.stdout.decode("utf-8"))

        while not Path(save_dir).exists():
            time.sleep(10)
