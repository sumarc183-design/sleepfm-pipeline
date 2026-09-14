"""Entraîne une sonde linéaire sur des embeddings figés pour la classification des stades."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LinearProbe(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_probe(
    probe: LinearProbe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> LinearProbe:
    probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        probe.train()
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(embeddings), labels)
            loss.backward()
            optimizer.step()

        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                preds = probe(embeddings).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"epoch {epoch + 1}/{epochs} — val accuracy: {correct / total:.4f}")

    return probe
