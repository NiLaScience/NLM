import argparse
import torch
from ..transformer import TransformerLM, generate, load_checkpoint
from ..tokenization.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to tokenizer vocab JSON")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to tokenizer merges TXT")
    
    # Generation
    parser.add_argument("--prompt", type=str, default="", help="Prompt text to continue from")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling threshold")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    
    # Device
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | mps. If None, auto-detect")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BPETokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
    eos_id = tokenizer.special_tokens.get("<|endoftext|>", None)
    
    # Load model from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_state = checkpoint["model_state_dict"]
    
    # Infer model config from state dict
    vocab_size = model_state["token_embedding.weight"].shape[0]
    d_model = model_state["token_embedding.weight"].shape[1]
    num_layers = len([k for k in model_state.keys() if k.startswith("layers.") and k.endswith(".weight")])
    
    # Create model with inferred config
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=2048,  # Default, can be made configurable
        d_model=d_model,
        num_layers=num_layers,
        num_heads=d_model // 64,  # Assuming d_k=64
        d_ff=None,  # Will use default
        rope_theta=10000.0,
        device=device,
    )
    model.load_state_dict(model_state)
    model.eval()
    
    # Encode prompt
    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt)
    else:
        prompt_ids = []
    
    print(f"\nPrompt: {args.prompt}")
    print("=" * 80)
    
    # Generate samples
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {i+1}:")
        
        output_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_tokens,
            eos_token_id=eos_id,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        
        # Decode and print
        generated_text = tokenizer.decode(output_ids.tolist())
        print(generated_text)
        
        if i < args.num_samples - 1:
            print("-" * 40)


if __name__ == "__main__":
    main()
