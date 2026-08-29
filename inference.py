import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from src.model import TinyCustomLLM

def run_test_generation(model, tokenizer, config, prompt="Eng", max_new_tokens=40, temperature=1.0, top_k=40, repetition_penalty=1.2):
    model.eval()
    encoded = tokenizer.encode(prompt)
    if not encoded.ids:
        # Fallback if prompt produces empty encoding
        input_ids = torch.tensor([[0]], dtype=torch.long)
        generated = [0]
    else:
        input_ids = torch.tensor([encoded.ids], dtype=torch.long)
        generated = input_ids.tolist()[0]
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -config['block_size']:]
            logits, _ = model(context)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            for token in set(generated):
                if next_token_logits[0, token] < 0:
                    next_token_logits[0, token] *= repetition_penalty
                else:
                    next_token_logits[0, token] /= repetition_penalty

            # Apply temperature safely
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                # Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
    return tokenizer.decode(generated)

def main():
    parser = argparse.ArgumentParser(description="Run inference on trained Kipsigis LLM")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer JSON file")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling limit")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt loop")
    args = parser.parse_args()

    print("-> Loading config...")
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"[FATAL] Config file not found at '{config_path}'!")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint_path = args.checkpoint or config.get("checkpoint_path", "checkpoints/kipsigis_model.pt")
    tokenizer_path = args.tokenizer or config.get("tokenizer_path", "tokenizer/kipsigis_tokenizer.json")

    if not os.path.exists(tokenizer_path):
        print(f"[FATAL] Tokenizer file not found at '{tokenizer_path}'!")
        print("Please train the model first by running 'python train.py'.")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"[FATAL] Model checkpoint file not found at '{checkpoint_path}'!")
        print("Please train the model first by running 'python train.py'.")
        sys.exit(1)

    print(f"-> Loading tokenizer from '{tokenizer_path}'...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = ByteLevel()

    print("-> Initializing model architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCustomLLM(config).to(device)

    print(f"-> Loading weights from '{checkpoint_path}'...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if args.interactive:
        print("\n=== Interactive Generation Mode (Ctrl+C to exit) ===")
        while True:
            try:
                user_prompt = input("\nEnter prompt: ").strip()
                if not user_prompt:
                    continue
                output = run_test_generation(
                    model, tokenizer, config,
                    prompt=user_prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                print(f"Generated: {output}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break
    else:
        prompts = [args.prompt] if args.prompt else ["Kamuktaindet", "Eng taunet", "Noa"]
        for prompt in prompts:
            output = run_test_generation(
                model, tokenizer, config,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            print(f"\nPrompt: '{prompt}'")
            print(f"Generated: {output}")

if __name__ == "__main__":
    main()
