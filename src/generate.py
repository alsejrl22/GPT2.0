import torch
import torch.nn.functional as F

from model import TinyGPT


@torch.no_grad()
def generate_text(model, block_size, stoi, itos, device, start_text, max_new_tokens):
    model.eval()

    context = torch.zeros((1, block_size), dtype=torch.long, device=device)

    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)

    out = list(start_text)

    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)

    return "".join(out)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint("checkpoints/tiny_gpt.pt", device)

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        emb_dim=checkpoint["emb_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    result = generate_text(
        model=model,
        block_size=checkpoint["block_size"],
        stoi=checkpoint["stoi"],
        itos=checkpoint["itos"],
        device=device,
        start_text="ROMEO:",
        max_new_tokens=500,
    )

    print(result)


if __name__ == "__main__":
    main()