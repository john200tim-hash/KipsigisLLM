import yaml
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from src.model import TinyCustomLLM

def run_test_generation(model, tokenizer, config, prompt="Eng", max_new_tokens=40, temperature=1.5, top_k=40, repetition_penalty=1.2):
    model.eval()
    encoded = tokenizer.encode(prompt)
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

            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
    return tokenizer.decode(generated)

def main():
    print("-> Loading config...")
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("-> Loading custom tokenizer...")
    tokenizer = Tokenizer.from_file("tokenizer/kipsigis_tokenizer.json")
    tokenizer.decoder = ByteLevel()

    print("-> Initializing model architecture...")
    model = TinyCustomLLM(config)

    print("-> Loading trained weights from checkpoints/kipsigis_model.pt...")
    model.load_state_dict(torch.load("checkpoints/kipsigis_model.pt", map_location=torch.device('cpu')))
    model.eval()

    prompts = ["Kamuktaindet", "Eng taunet", "Noa"]
    
    for prompt in prompts:
        output = run_test_generation(model, tokenizer, config, prompt=prompt)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {output}")

if __name__ == "__main__":
    main()
