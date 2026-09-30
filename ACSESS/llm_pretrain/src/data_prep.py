import argparse
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
import torch
from multiprocessing import Pool
from tqdm import tqdm


def tokenize_chunk(args_tuple):
    """Tokenize a chunk of text using a tokenizer."""
    text, tokenizer_path = args_tuple
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokens = tokenizer.encode(text).ids
    return tokens


class DataPrep:
    def __init__(self, tokenizer_path: str, output_dir: str, vocab_size: int = 50304):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.endoftext_token = self.tokenizer.encode("<|endoftext|>").ids[0]

    def stream_dataset(self, dataset_name: str, config: str, split: str, weight: float, total_tokens: int):
        """Stream dataset and yield texts."""
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
        target_tokens = int(total_tokens * weight)
        tokens_seen = 0

        for example in ds:
            if tokens_seen >= target_tokens:
                break

            if isinstance(example, dict):
                text = example.get("text", "")
            else:
                text = str(example)

            if text:
                yield text
                tokens_seen += len(text.split())

    def tokenize_and_save(self, dataset_name: str, config: str, split: str, weight: float, total_tokens: int, is_val: bool = False):
        """Tokenize dataset and save as memmap shards."""
        print(f"Processing {dataset_name}/{config} (weight={weight})")

        all_tokens = []
        doc_count = 0

        for text in tqdm(self.stream_dataset(dataset_name, config, split, weight, total_tokens)):
            tokens = self.tokenizer.encode(text).ids
            all_tokens.extend(tokens)
            all_tokens.append(self.endoftext_token)
            doc_count += 1

        print(f"Tokenized {doc_count} documents, {len(all_tokens)} tokens total")

        if is_val:
            shard_name = f"val_{dataset_name.split('/')[-1]}"
        else:
            shard_name = f"shard_{dataset_name.split('/')[-1]}"

        shard_path = self.output_dir / f"{shard_name}.pt"
        tokens_array = torch.tensor(all_tokens, dtype=torch.uint16)
        torch.save(tokens_array, shard_path)
        print(f"Saved {len(all_tokens)} tokens to {shard_path}")

        return len(all_tokens)

    def prepare_mixture(self, mixture: list, total_tokens: int = int(25e9), val_ratio: float = 0.01):
        """Prepare full dataset mixture."""
        val_tokens = int(total_tokens * val_ratio)
        train_tokens = total_tokens - val_tokens

        tokens_per_source = {}
        total_train_tokens = 0

        for source in mixture:
            name = source["dataset"]
            config = source.get("config", "default")
            weight = source.get("weight", 1.0)

            tokens = self.tokenize_and_save(name, config, "train", weight, train_tokens, is_val=False)
            tokens_per_source[name] = tokens
            total_train_tokens += tokens

            val_tokens_src = self.tokenize_and_save(name, config, "train", weight * val_ratio / (1 - val_ratio), train_tokens, is_val=True)

        print(f"\n=== Summary ===")
        print(f"Total training tokens: {total_train_tokens:,}")
        print(f"Tokens per source:")
        for name, count in tokens_per_source.items():
            print(f"  {name}: {count:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_tokens", type=int, default=int(25e9))
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--no_translation", action="store_true", help="Skip German translation data")
    args = parser.parse_args()

    if args.no_translation:
        mixture = [
            {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-100BT", "weight": 0.60},
            {"dataset": "mlfoundations/dclm-baseline-1.0", "config": "default", "weight": 0.27},
            {"dataset": "bigcode/starcoderdata", "config": "default", "weight": 0.08},
            {"dataset": "HuggingFaceTB/finemath", "config": "finemath-3plus", "weight": 0.05},
        ]
    else:
        mixture = [
            {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-100BT", "weight": 0.55},
            {"dataset": "mlfoundations/dclm-baseline-1.0", "config": "default", "weight": 0.22},
            {"dataset": "HuggingFaceFW/fineweb-2", "config": "deu_Latn", "weight": 0.10},
            {"dataset": "bigcode/starcoderdata", "config": "default", "weight": 0.08},
            {"dataset": "HuggingFaceTB/finemath", "config": "finemath-3plus", "weight": 0.05},
        ]

    prep = DataPrep(args.tokenizer_path, args.output_dir, args.vocab_size)
    prep.prepare_mixture(mixture, args.total_tokens)


if __name__ == "__main__":
    main()
