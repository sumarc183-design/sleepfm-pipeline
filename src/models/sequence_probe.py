"""Sonde séquentielle : BiGRU sur la suite des embeddings SleepFM d'une nuit.

Les autres sondes (linear_probe.py) classent chaque époque de 30 s isolément.
Or le stade d'une époque dépend fortement de ses voisines (on ne passe pas
de N3 à REM en 30 s, N1 se reconnaît surtout aux transitions) : un expert
lit l'hypnogramme en contexte. Ce module garde les embeddings figés et
apprend seulement une petite tête récurrente bidirectionnelle par-dessus,
qui voit pour chaque époque celles d'avant et d'après.

Interface proche de sklearn, mais fit/predict prennent en plus le sujet et
le numéro d'époque de chaque ligne (`groups`), pour reconstituer les nuits
dans l'ordre chronologique.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _BiGRUTagger(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classify = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, dim) -> (batch, time, classes)
        out, _ = self.gru(self.project(x))
        return self.classify(out)


def split_into_nights(subjects: np.ndarray, epoch_idx: np.ndarray) -> list[np.ndarray]:
    """Indices de lignes de chaque nuit, triés par numéro d'époque."""
    nights = []
    for subject in dict.fromkeys(subjects):  # ordre d'apparition, déterministe
        rows = np.flatnonzero(subjects == subject)
        nights.append(rows[np.argsort(epoch_idx[rows], kind="stable")])
    return nights


class SequenceProbe:
    needs_groups = True  # predict/predict_proba prennent (sujets, numéros d'époque)

    def __init__(
        self,
        hidden_dim: int = 128,
        epochs: int = 25,
        lr: float = 1e-3,
        dropout: float = 0.3,
        window: int = 120,
        batch_size: int = 16,
        device: str = "cpu",
        seed: int = 42,
    ):
        # window = 120 époques = 1 h de contexte par exemple d'entraînement ;
        # à la prédiction, le BiGRU lit la nuit entière.
        self.hidden_dim, self.epochs, self.lr, self.dropout = hidden_dim, epochs, lr, dropout
        self.window, self.batch_size, self.device, self.seed = window, batch_size, device, seed

    def _windows(self, X: np.ndarray, y: np.ndarray, nights: list[np.ndarray], rng: np.random.Generator):
        """Découpe chaque nuit en fenêtres consécutives, avec un décalage
        aléatoire à chaque passe pour varier les frontières."""
        windows = []
        for rows in nights:
            first_cut = int(rng.integers(1, self.window + 1))  # 1..window : première fenêtre raccourcie
            bounds = [0, *range(first_cut, len(rows), self.window), len(rows)]
            windows += [rows[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]
        rng.shuffle(windows)
        for i in range(0, len(windows), self.batch_size):
            batch = windows[i : i + self.batch_size]
            length = max(len(w) for w in batch)
            xb = np.zeros((len(batch), length, X.shape[1]), dtype=np.float32)
            yb = np.full((len(batch), length), -100, dtype=np.int64)  # -100 = rembourrage ignoré par la loss
            for j, w in enumerate(batch):
                xb[j, : len(w)] = X[w]
                yb[j, : len(w)] = y[w]
            yield torch.from_numpy(xb).to(self.device), torch.from_numpy(yb).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: tuple[np.ndarray, np.ndarray]) -> SequenceProbe:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        nights = split_into_nights(*groups)

        # Arrêt précoce sur ~10 % des nuits du TRAIN (jamais le test).
        order = rng.permutation(len(nights))
        n_val = max(1, len(nights) // 10) if len(nights) >= 5 else 0
        val_nights = [nights[i] for i in order[:n_val]]
        train_nights = [nights[i] for i in order[n_val:]]

        self.model_ = _BiGRUTagger(X.shape[1], self.hidden_dim, len(self.classes_), self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        best_loss, best_state = float("inf"), None
        for _ in range(self.epochs):
            self.model_.train()
            for xb, yb in self._windows(X, y_encoded, train_nights, rng):
                optimizer.zero_grad()
                loss = criterion(self.model_(xb).flatten(0, 1), yb.flatten())
                loss.backward()
                optimizer.step()
            if val_nights:
                val_loss = self._loss(X, y_encoded, val_nights, criterion)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    @torch.no_grad()
    def _loss(self, X, y_encoded, nights, criterion) -> float:
        self.model_.eval()
        losses = [
            criterion(self._logits(X[rows]), torch.from_numpy(y_encoded[rows]).to(self.device)).item()
            for rows in nights
        ]
        return float(np.mean(losses))

    def _logits(self, X_night: np.ndarray) -> torch.Tensor:
        return self.model_(torch.from_numpy(X_night.astype(np.float32))[None].to(self.device))[0]

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, groups: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        proba = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        for rows in split_into_nights(*groups):  # nuit entière d'un coup : contexte complet
            proba[rows] = torch.softmax(self._logits(X[rows]), dim=-1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray, groups: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return self.classes_[self.predict_proba(X, groups).argmax(axis=1)]
