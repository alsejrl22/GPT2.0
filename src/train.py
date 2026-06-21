from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import prepare_data
from model import TinyGPT


block_size = 64
batch_size = 64
emb_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.1
learning_rate = 3e-4
epochs = 5
max_steps = 300


def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

        if step + 1 >= max_steps:
            break

    return total_loss / total_count


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, tokenizer = prepare_data(block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Tiny GPT training start")
    print(f"device: {device}")
    print(f"vocab_size: {tokenizer.vocab_size}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch {epoch + 1} | train loss {train_loss:.4f}")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "vocab_size": tokenizer.vocab_size,
            "block_size": block_size,
            "emb_dim": emb_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        checkpoint_dir / "tiny_gpt.pt",
    )

    print("model saved")


if __name__ == "__main__":
    main()