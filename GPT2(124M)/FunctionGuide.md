# Guide to Project Code

## Table of Contents
1. [File Structure](#file-structure)
2. [pre_trained_gpt2.py](#pre_trained_gpt2py)
3. [train_gpt2.py](#train_gpt2py)
4. [Fineweb-Edu Dataset Preprocessing Script](#fineweb-edu-dataset-preprocessing-script)
    - [Script Overview](#script-overview)
    - [Detailed Explanation](#detailed-explanation)
    - [Usage](#usage)
5. [hellswag.py](#hellswagpy)
6. [play.ipynb](#playipynb)

## File Structure
```
GPT2(124M)
│   README.md
│   FunctionGuide.md
│   pre_trained_gpt2.py
│   train_gpt2.py
│   fineweb.py
│   hellswag.py
│   play.ipynb
```

## FineWeb-Edu Dataset Preprocessing Script

The `fineweb.py` script is designed to download, tokenize, and save data shards of the FineWeb-Edu dataset for training models using the `datasets` library from HuggingFace. Below is a detailed explanation of the script:

### Script Overview

The script performs the following tasks:
1. **Create Local Directory**: Creates a local directory to save the data shards if it doesn't already exist.
2. **Download Dataset**: Downloads the FineWeb-Edu dataset from HuggingFace.
3. **Initialize Tokenizer**: Initializes the tokenizer using the `tiktoken` library with GPT-2 encoding.
4. **Tokenize Documents**: Tokenizes the dataset documents.
5. **Save Data Shards**: Saves the tokenized data into shards of a specified size.

### Detailed Explanation

#### Imports and Constants

The script imports necessary libraries and sets constants:
- `os`: For directory and file operations.
- `multiprocessing` as `mp`: For parallel processing.
- `numpy` as `np`: For handling numerical operations.
- `tiktoken`: For tokenization.
- `datasets`: For loading the dataset.
- `tqdm`: For progress bars.

Constants are defined for local directory, remote dataset name, and shard size.

```
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards
```

#### Create Local Directory

Creates the local directory to store data shards if it does not already exist.

```
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
```

#### Download Dataset

Downloads the FineWeb-Edu dataset from HuggingFace.

```
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
```

#### Initialize Tokenizer

Initializes the GPT-2 tokenizer from `tiktoken`.

```
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['']  # end of text token
```

#### Tokenize Documents

Defines a function to tokenize a single document and return a numpy array of uint16 tokens.

```
def tokenize(doc):
    tokens = [eot]  # the special token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16
```

#### Save Data Shards

Defines a function to save the tokenized data to a file.

```
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)
```

#### Process and Save Shards

Uses multiprocessing to tokenize all documents and write the tokenized data into shards. Each shard has a size of `shard_size` tokens. The last shard may contain fewer tokens if the total number of tokens is not a multiple of `shard_size`.

```
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
```

### Usage

To run the script, simply execute it as a Python file:

```
$ python fineweb.py
```

This will download the FineWeb-Edu dataset, tokenize it, and save the tokenized data in shards in the local directory `edu_fineweb10B`.

## HellaSwag Dataset Evaluation Script

This script downloads, tokenizes, and evaluates the HellaSwag dataset using a GPT-2 model. It provides a comprehensive way to measure the performance of different model variants on the HellaSwag task. Below is a detailed explanation of the script:

### Script Overview

The script performs the following tasks:
1. **Download HellaSwag Dataset**: Downloads the dataset files from the specified URLs.
2. **Initialize Tokenizer**: Initializes the tokenizer using the `tiktoken` library with GPT-2 encoding.
3. **Render Examples**: Prepares dataset examples for evaluation by tokenizing the context and candidate endings.
4. **Evaluate Model**: Evaluates the performance of a specified GPT-2 model variant on the validation set.

### Detailed Explanation

#### Imports and Constants

The script imports necessary libraries and sets constants:
- `os`, `json`, `requests`: For file and HTTP operations.
- `tiktoken`: For tokenization.
- `tqdm`: For progress bars.
- `torch`, `torch.nn`, `torch.nn.functional`, `transformers`: For model evaluation.

Constants are defined for the data cache directory and dataset URLs.

```
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}
```

#### Download Helper Function

Defines a helper function to download files from a given URL.

```
def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
```

#### Download Dataset

Downloads the specified split of the HellaSwag dataset.

```
def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
```

#### Render Examples

Prepares the examples from the dataset for evaluation by tokenizing the context and endings.

```
def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label
```

#### Iterate Over Examples

Iterates over the examples in the specified split of the dataset.

```
def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example
```

#### Model Evaluation

Evaluates the specified GPT-2 model variant on the HellaSwag validation set.

```
@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for I, end in enumerate(example["endings"]):
                print(f"{I} (loss: {avg_loss[I].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")
```

#### Main Execution

Defines the command-line interface for specifying model type and device, and runs the evaluation.

```
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
```

### Usage

To run the script, execute it from the command line with the desired model type and device:

```
$ python hellaswag.py -m gpt2 -d cuda
```

This will download the HellaSwag dataset, prepare the examples, and evaluate the specified model variant on the validation set, providing accuracy metrics.
