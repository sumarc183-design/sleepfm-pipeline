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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class _BiGRUTagger(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classify = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """(batch, time, dim) -> (batch, time, classes). `lengths` = longueur
        réelle de chaque séquence d'un lot complété par des zéros : le GRU
        s'arrête à la fin de chacune. Sans elle, le sens retour démarrait sur
        le rembourrage, qui influençait alors les vraies époques."""
        h = self.project(x)
        if lengths is None:
            out, _ = self.gru(h)
        else:
            packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=x.shape[1])
        return self.classify(out)


EPOCH_SECONDS = 30.0
MAX_STEP_SECONDS = EPOCH_SECONDS + 1.0  # au-delà : trou d'enregistrement


def split_into_nights(
    subjects: np.ndarray, epoch_idx: np.ndarray, onset_s: np.ndarray | None = None
) -> list[np.ndarray]:
    """Indices de lignes de chaque segment continu, triés par numéro d'époque.
    Sans temps d'origine (`onset_s` absent ou entièrement NaN), une nuit = un
    segment. Un temps manquant isolé est refusé : il masquerait les trous.
    Avec, la nuit est coupée là où deux époques consécutives sont séparées de
    plus de 31 s : le BiGRU ne relie pas deux époques de part et d'autre d'une
    période retirée (non annotée)."""
    segments = []
    for subject in dict.fromkeys(subjects):  # ordre d'apparition, déterministe
        rows = np.flatnonzero(subjects == subject)
        rows = rows[np.argsort(epoch_idx[rows], kind="stable")]
        if onset_s is None or np.isnan(onset_s[rows]).all():
            segments.append(rows)
            continue
        if not np.isfinite(onset_s[rows]).all():
            raise ValueError(f"{subject} : temps d'origine partiellement manquants ou infinis")
        breaks = np.flatnonzero(np.diff(onset_s[rows]) > MAX_STEP_SECONDS) + 1
        segments += [part for part in np.split(rows, breaks) if len(part)]
    return segments


class SequenceProbe:
    needs_groups = True  # predict/predict_proba prennent (sujets, numéros d'époque[, débuts en s])

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
        context: int | None = None,
        class_weight: str | None = None,
    ):
        # window = 120 époques = 1 h de contexte par exemple d'entraînement ;
        # à la prédiction, le BiGRU lit la nuit entière (context=None), ou des
        # blocs consécutifs de `context` époques (ablation du contexte).
        self.hidden_dim, self.epochs, self.lr, self.dropout = hidden_dim, epochs, lr, dropout
        self.window, self.batch_size, self.device, self.seed = window, batch_size, device, seed
        self.context = context
        # "balanced" : poids de chaque stade inversement proportionnel à sa
        # fréquence dans le train, comme la régression logistique officielle.
        self.class_weight = class_weight

    def get_params(self) -> dict:
        keys = ("hidden_dim", "epochs", "lr", "dropout", "window", "batch_size", "seed", "context", "class_weight")
        return {key: getattr(self, key) for key in keys}

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
            lengths = torch.tensor([len(w) for w in batch])
            yield torch.from_numpy(xb).to(self.device), torch.from_numpy(yb).to(self.device), lengths

    def fit(self, X: np.ndarray, y: np.ndarray, groups: tuple) -> SequenceProbe:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        segments = split_into_nights(*groups)

        # Arrêt précoce sur ~10 % des nuits du TRAIN (jamais le test), tirées
        # par sujet : tous les segments d'une nuit vont du même côté.
        subjects = list(dict.fromkeys(groups[0]))
        order = rng.permutation(len(subjects))
        n_val = max(1, len(subjects) // 10) if len(subjects) >= 5 else 0
        by_subject: dict = {}
        for rows in segments:
            by_subject.setdefault(groups[0][rows[0]], []).append(rows)
        val_nights = [rows for i in order[:n_val] for rows in by_subject[subjects[i]]]
        train_nights = [rows for i in order[n_val:] for rows in by_subject[subjects[i]]]

        self.model_ = _BiGRUTagger(X.shape[1], self.hidden_dim, len(self.classes_), self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-2)
        weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_encoded, minlength=len(self.classes_))
            weight = torch.tensor(len(y_encoded) / (len(self.classes_) * np.maximum(counts, 1)), dtype=torch.float32)
            weight = weight.to(self.device)
        elif self.class_weight is not None:
            raise ValueError(f"class_weight inconnu : {self.class_weight!r}")
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        best_loss, best_state, best_epoch = float("inf"), None, None
        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb, lengths in self._windows(X, y_encoded, train_nights, rng):
                optimizer.zero_grad()
                loss = criterion(self.model_(xb, lengths).flatten(0, 1), yb.flatten())
                loss.backward()
                optimizer.step()
            if val_nights:
                val_loss = self._loss(X, y_encoded, val_nights, criterion)
                if val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch + 1
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.fit_info_ = {
            "validation_subjects": sorted(str(subjects[i]) for i in order[:n_val]),
            "best_epoch": best_epoch,
            "best_validation_loss": None if best_state is None else float(best_loss),
        }
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
    def predict_proba(self, X: np.ndarray, groups: tuple) -> np.ndarray:
        proba = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        for night in split_into_nights(*groups):
            step = self.context or len(night)  # par défaut, nuit entière d'un coup
            for start in range(0, len(night), step):
                rows = night[start : start + step]
                proba[rows] = torch.softmax(self._logits(X[rows]), dim=-1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray, groups: tuple) -> np.ndarray:
        return self.classes_[self.predict_proba(X, groups).argmax(axis=1)]
