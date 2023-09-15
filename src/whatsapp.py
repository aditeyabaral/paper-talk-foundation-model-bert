import argparse
import json
import logging
import random
import re
from pathlib import Path

from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    filename="whatsapp.log",
    level=logging.DEBUG,
    filemode="w",
)

parser = argparse.ArgumentParser("Parse WhatsApp chat history")
parser.add_argument(
    "--input", "-i", type=str, required=True, help="Input directory"
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    required=True,
    help="Output directory. If not specified, will be saved in same directory as input file",
)
parser.add_argument(
    "--combine", action="store_true", help="Combine successive messages from same user"
)
args = parser.parse_args()


def process_file(filepath: Path):
    with open(filepath, "r", encoding="utf8") as f:
        content = f.read()

    # 28/06/2023, 10:10 pm
    datetime_format = r'\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2} [apAP][mM] - '
    lines = re.split(f"{datetime_format}", content)

    group_name = filepath.stem
    members = set()
    messages = list()
    for line in tqdm(lines):
        line = line.strip()
        if not line or "<Media omitted>" in line or ": " not in line:
            continue
        try:
            author, message = line.split(":", 1)
            author = author.strip()
            message = message.strip()
            messages.append({
                "sender": author,
                "sender_id": author,
                "recipient": group_name,
                "recipient_id": group_name,
                "message": message,
            })
            members.add(author)
        except Exception as e:
            logging.error(f"Error parsing line: {line}\nError: {e}")
            continue

    members = list(members)
    if len(members) == 2:
        group_type = "direct"
    else:
        group_type = "group"

    return group_name, group_type, members, messages


def process_chats():
    messages = list()
    members = {"names": [], "ids": []}
    groups = dict()

    filenames = list(Path(args.input).glob("*.txt"))
    for filepath in tqdm(filenames):
        group_name, group_type, group_members, group_messages = process_file(filepath)
        groups[group_name] = {
            "title": group_name,
            "member_names": group_members,
            "member_ids": group_members,
            "type": group_type,
        }
        members["names"].extend(group_members)
        members["ids"].extend(group_members)
        messages.extend(group_messages)

    members["names"] = list(set(members["names"]))
    members["ids"] = list(set(members["ids"]))
    random.shuffle(messages)
    return messages, members, groups


def combine_successive_messages(author_list, message_list):
    author_list_new = list()
    message_list_new = list()
    len_author_list = len(author_list)
    for i in tqdm(range(len_author_list)):
        if i == 0:
            author_list_new.append(author_list[i])
            message_list_new.append(message_list[i])
        else:
            if author_list[i] == author_list[i - 1]:
                message_list_new[-1] = message_list_new[-1] + "\n" + message_list[i]
            else:
                author_list_new.append(author_list[i])
                message_list_new.append(message_list[i])

    return author_list_new, message_list_new


args.input = Path(args.input).absolute()
args.output = Path(args.output).absolute()
logging.info(f"Arguments: {args}")

if not Path(args.input).exists():
    logging.error(f"File {args.input} does not exist")
    raise FileNotFoundError(f"File {args.input} does not exist")

if not Path(args.output).parent.exists():
    logging.error(f"Directory {Path(args.output).parent} does not exist")
    raise FileNotFoundError(f"Directory {Path(args.output).parent} does not exist")

logging.info(f"Processing files in {args.input} and saving to {args.output}")
messages, members, groups = process_chats()
logging.info(f"Found {len(messages)} messages, {len(members['names'])} members, {len(groups)} groups")

with open(args.output / "messages.json", "w", encoding="utf8") as f:
    json.dump(messages, f, indent=4, ensure_ascii=False)
with open(args.output / "members.json", "w", encoding="utf8") as f:
    json.dump(members, f, indent=4, ensure_ascii=False)
with open(args.output / "groups.json", "w", encoding="utf8") as f:
    json.dump(groups, f, indent=4, ensure_ascii=False)
