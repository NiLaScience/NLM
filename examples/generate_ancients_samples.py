#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

from NLM.min_lm.tokenization.bpe import BPETokenizer
from NLM.min_lm.transformer import TransformerLM, generate

EOT = "<|endoftext|>"


def main():
    parser = argparse.ArgumentParser(description="Generate samples from ancients-continued model")
    parser.add_argument("--checkpoint", type=str, default=str(Path("NLM/checkpoints/ancients_model.pt")))
    parser.add_argument("--vocab_path", type=str, default=str(Path("NLM/tokenizers/tinystories_vocab.json")))
    parser.add_argument("--merges_path", type=str, default=str(Path("NLM/tokenizers/tinystories_merges.txt")))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
        device=device,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=[EOT])
    eos_id = tokenizer.special_tokens.get(EOT, None)

    prompts = [
        "Sing, goddess, of the wrath of Achilles",
        "O Muse, tell me of the man of many ways",
        "It was the habit of the philosopher to say",
        "When the consul had taken the field, the senate decreed",
        "Thus spoke the king to his captains",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Prompt: {prompt}")
        prompt_ids = tokenizer.encode(prompt)
        out_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print(tokenizer.decode(out_ids.tolist()))


if __name__ == "__main__":
    main()


