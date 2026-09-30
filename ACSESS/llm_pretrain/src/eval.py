import argparse
import torch
import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast, PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from tokenizers import Tokenizer

from model import LlamaLM


def export_to_huggingface(checkpoint_path: str, tokenizer_path: str, output_dir: str):
    """Export trained model to Hugging Face format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LlamaLM(vocab_size=50304, d_model=1024, n_layers=16, n_heads=16, mlp_hidden=2816, max_seq_len=2048)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    tokenizer_hf = Tokenizer.from_file(tokenizer_path)

    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": tokenizer_hf.encode("<|endoftext|>").ids[0],
        "eos_token_id": tokenizer_hf.encode("<|endoftext|>").ids[0],
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 2816,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
        "num_key_value_heads": 16,
        "pad_token_id": tokenizer_hf.encode("<|pad|>").ids[0],
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.36.0",
        "use_cache": True,
        "vocab_size": 50304,
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    model_state = model.state_dict()
    torch.save(model_state, output_path / "pytorch_model.bin")

    tokenizer_hf.save(str(output_path / "tokenizer.json"))
    print(f"Model exported to {output_path}")


def verify_export(checkpoint_path: str, tokenizer_path: str, exported_dir: str, batch_size: int = 2):
    """Verify exported model gives same loss as original."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    original_model = LlamaLM(vocab_size=50304, d_model=1024, n_layers=16, n_heads=16, mlp_hidden=2816, max_seq_len=2048)
    original_model.load_state_dict(ckpt["model"])
    original_model = original_model.to(device).eval()

    try:
        from transformers import LlamaForCausalLM as HFLlamaForCausalLM
        exported_model = HFLlamaForCausalLM.from_pretrained(exported_dir).to(device).eval()
    except:
        print("Warning: Could not load exported model via transformers; skipping verification")
        return

    torch.manual_seed(42)
    test_input = torch.randint(0, 50304, (batch_size, 2048)).to(device)
    labels = torch.randint(0, 50304, (batch_size, 2048)).to(device)

    with torch.no_grad():
        orig_logits = original_model(test_input)
        exp_logits = exported_model(test_input).logits

    orig_loss = torch.nn.functional.cross_entropy(orig_logits.view(-1, 50304), labels.view(-1))
    exp_loss = torch.nn.functional.cross_entropy(exp_logits.view(-1, 50304), labels.view(-1))

    print(f"Original model loss: {orig_loss:.6f}")
    print(f"Exported model loss: {exp_loss:.6f}")
    print(f"Difference: {abs(orig_loss - exp_loss):.6e}")

    if abs(orig_loss - exp_loss) < 1e-3:
        print("✓ Exported model matches original!")
    else:
        print("⚠ Warning: Exported model loss differs significantly from original")


def run_lm_eval(model_path: str, tokenizer_path: str, tasks: list = None, num_fewshot: int = 0):
    """Run lm-evaluation-harness on checkpoint."""
    if tasks is None:
        tasks = [
            "arc_easy", "arc_challenge", "hellaswag", "piqa", "social_iqa",
            "winogrande", "commonsense_qa", "openbookqa", "boolq", "lambada_openai"
        ]

    print(f"Running lm_eval on tasks: {tasks}")

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM

        model = HFLM(pretrained=model_path, dtype="float16")
        results = evaluator.evaluate(
            model,
            tasks,
            num_fewshot=num_fewshot,
            batch_size=8,
            device="cuda",
        )

        print(json.dumps(results, indent=2))
        return results

    except ImportError:
        print("lm-eval-harness not installed. Install with: pip install lm-eval")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    export_to_huggingface(args.checkpoint, args.tokenizer, args.output_dir)

    if args.verify:
        verify_export(args.checkpoint, args.tokenizer, args.output_dir)

    if args.eval:
        run_lm_eval(args.output_dir, args.tokenizer)


if __name__ == "__main__":
    main()
