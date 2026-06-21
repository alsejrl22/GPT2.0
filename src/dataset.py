from pathlib import Path
import urllib.request
import torch
from torch.utils.data import Dataset


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tiny_shakespeare():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "input.txt"

    if not file_path.exists():
        urllib.request.urlretrieve(DATA_URL, file_path)

    return file_path


def load_text():
    file_path = download_tiny_shakespeare()
    return file_path.read_text(encoding="utf-8")


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])


class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def prepare_data(block_size):
    text = load_text()
    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    dataset = NextTokenDataset(encoded, block_size)
    return dataset, tokenizer