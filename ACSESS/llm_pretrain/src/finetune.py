import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from model import LlamaLM


class TextToTextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    prompt, target = parts[0], parts[1]
                    self.examples.append((prompt, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, target = self.examples[idx]
        tokens = self.tokenizer.encode(prompt).ids + self.tokenizer.encode(target).ids
        tokens = tokens[:self.max_length - 1]
        tokens.append(self.tokenizer.encode("<|endoftext|>").ids[0])

        prompt_len = len(self.tokenizer.encode(prompt).ids)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        labels[:prompt_len - 1] = -100

        return input_ids, labels


def finetune(checkpoint_path: str, train_data_path: str, tokenizer_path: str, output_dir: str, num_epochs: int = 3):
    """Fine-tune pretrained model on downstream task."""
    from tokenizers import Tokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LlamaLM(vocab_size=50304, d_model=1024, n_layers=16, n_heads=16, mlp_hidden=2816, max_seq_len=2048)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    dataset = TextToTextDataset(train_data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    for epoch in range(num_epochs):
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, 50304), labels.view(-1), reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Fine-tuned model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    finetune(args.checkpoint, args.train_data, args.tokenizer, args.output_dir, args.num_epochs)


if __name__ == "__main__":
    main()
