"""
CLI chat interface for a trained liteGPT checkpoint.

Usage:
    python -m litegpt.chat --checkpoint checkpoints/final.pt
    python -m litegpt.chat --checkpoint checkpoints/final.pt --device mps
"""

import argparse
import torch
from litegpt.model import GPT, GPTConfig
from litegpt.tokenizer import Tokenizer


def load_checkpoint(path: str, device: torch.device) -> tuple[GPT, GPTConfig]:
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model  = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.setup_rope()
    model.eval()
    return model, config


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str,   required=True)
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--max_new_tokens",  type=int,   default=200)
    parser.add_argument("--temperature",     type=float, default=0.8)
    parser.add_argument("--top_k",           type=int,   default=50)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    print(f"Loading {args.checkpoint} on {device}...")
    model, config = load_checkpoint(args.checkpoint, device)
    tokenizer = Tokenizer()

    n_params = model.num_parameters()
    print(f"Model ready — {n_params/1e6:.1f}M params | depth={config.depth}\n")
    print("Type a message and press Enter. Ctrl+C to quit.\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})

        # Build prompt tokens — append assistant_start so model continues
        prompt_ids = tokenizer.encode_chat(history) + [tokenizer.asst_s]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        new_ids  = out[0, len(prompt_ids):].tolist()
        response = tokenizer.decode_response(new_ids)

        print(f"Assistant: {response}\n")
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
