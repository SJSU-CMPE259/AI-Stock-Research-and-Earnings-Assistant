import argparse
import tempfile
import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def iter_dataset_sample(dataset_name: str, config: str, split: str, sample_tokens: int = 5e9, langs: list = None):
    """Iterate through dataset sampling up to sample_tokens."""
    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    tokens_count = 0

    for example in ds:
        if isinstance(example, dict):
            text = example.get("text", "")
        else:
            text = str(example)

        if langs and hasattr(example, "language"):
            if example["language"] not in langs:
                continue

        if text:
            yield text
            tokens_count += len(text.split())

        if tokens_count >= sample_tokens:
            break


def get_training_corpus(mixture: dict, total_tokens: int = 5e9):
    """Iterate through mixed dataset."""
    for source in mixture:
        name = source["dataset"]
        config = source.get("config", "default")
        split = source.get("split", "train")
        weight = source.get("weight", 1.0)
        langs = source.get("langs", None)

        source_tokens = int(total_tokens * weight)
        print(f"Sampling {source_tokens/1e9:.1f}B tokens from {name}/{config}")

        for text in iter_dataset_sample(name, config, split, source_tokens, langs):
            yield text


def train_tokenizer(output_path: str, vocab_size: int = 50304, mixture: dict = None):
    """Train a BPE tokenizer on a mixed dataset."""

    if mixture is None:
        mixture = [
            {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-100BT", "weight": 0.55},
            {"dataset": "mlfoundations/dclm-baseline-1.0", "config": "default", "weight": 0.22},
            {"dataset": "HuggingFaceFW/fineweb-2", "config": "deu_Latn", "weight": 0.10, "langs": ["de"]},
            {"dataset": "bigcode/starcoderdata", "config": "default", "weight": 0.08},
            {"dataset": "HuggingFaceTB/finemath", "config": "finemath-3plus", "weight": 0.05},
        ]

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Lowercase(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|unk|>", "<|endoftext|>", "<|pad|>"],
        show_progress=True,
    )

    print("Training BPE tokenizer...")
    corpus = get_training_corpus(mixture, total_tokens=int(7.5e9))

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Path to save tokenizer")
    parser.add_argument("--vocab_size", type=int, default=50304)
    args = parser.parse_args()

    tokenizer = train_tokenizer(args.output_path, args.vocab_size)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
