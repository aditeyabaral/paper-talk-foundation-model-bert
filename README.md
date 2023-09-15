# Research et al. - Paper Talk Episode 4 - Building Foundation Models using BERT

This repository contains the code for the hands-on session of the paper talk on building foundation models using BERT.

## Setup

1. Clone the repository
2. Create a virtual environment using conda (recommended) or venv and activate it. Then, install the dependencies using
   the following command:
    ```bash
    conda create -n bert python=3.8
    conda activate bert
    pip install -r requirements.txt
    ```
3. Set up HuggingFace code repository
    1. Visit [HuggingFace](https://huggingface.co/)'s website and create an account. Remember your username!
    2. Navigate to [Access Tokens](https://huggingface.co/settings/tokens) and generate a token with **write access**.
    3. Copy your username and token to `src/train_tokenizer.py` (L13, L15)
4. Training on your data (optional) - a data directory consisting of exported chat history from the `PESU26`, `PESU27`
   and `Tech Talks` groups is provided in the `data` directory. If you wish to train on your own data, follow the steps
   below:
    1. Export your chat history from multiple groups WhatsApp. You can do this by opening the chat, clicking on the
       three dots in the top right corner and selecting `More > Export chat`. Select `Without Media` and export the
       chat history. Store all the `.txt` files in a directory.
    2. You need to now convert the `.txt` files into `.json` files which the trainer can use. To do this, run the
       following command:
        ```bash
        python src/whatsapp.py --input <path to directory containing .txt files> --output_dir <path to output directory>
        ``` 
    3. You can now train the tokenizer on your data by running the following command. Modify L7
       in `src/train_tokenizer.py` to add the path to the newly generated `messages.json` file and then run the script.
        ```bash
        python src/train_tokenizer.py
        ```
       After running it, you should verify the two things:
        1. A new directory called `tokenizer` has been created in the current working directory.
        2. Visit `https://huggingface.co/<your_username>/tokenizer`. You should be able to see this repository that
           holds your new tokenizer
       3. 
