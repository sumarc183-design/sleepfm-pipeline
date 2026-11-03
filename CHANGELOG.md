# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [0.1.0] — 2026-11-03

Premier jalon : baseline SleepFM reproduite de bout en bout sur PhysioNet 2018.

### Ajouté
- `run_baseline.py`, point d'entrée unique : split par sujet, extraction des
  époques, embeddings SleepFM figés (checkpoint officiel `best.pt`), sondes,
  évaluation, export JSON des métriques (`--results-json`).
- Mode par lots (`--physionet-subjects`, `--embeddings-dir`) : téléchargement
  au fil de l'eau, cache d'un `.npz` d'embeddings par sujet, reprise après
  interruption.
- Sondes comparées sur les mêmes embeddings : régression logistique
  (baseline), logistique pondérée, MLP, XGBoost, BiGRU sur la nuit.
- Sélection MLP / XGBoost par validation croisée par sujet sur le train,
  avec élimination en cas de surapprentissage ; test évalué une seule fois.
- Notebook Google Colab (GPU gratuit) avec cache sur Google Drive.
- Résultats : κ = 0,622, accuracy 0,722, F1 macro 0,681 sur 100 sujets
  (75 train / 25 test).
- CI (lint ruff puis tests), `pyproject.toml`, `pre-commit`, carte du modèle,
  guide d'utilisation.

### Corrigé
- Chargement du `best.pt` publié, dont les noms de clés diffèrent de ceux
  attendus par le script officiel.
- Commandes de démarrage du README (dossier de sortie du téléchargement).

### Retiré
- Random Forest (plus lent que XGBoost, moins adapté aux embeddings denses).
- `scripts/run_embedding_and_probe.py` et `src/eval/evaluate.py`, doublons de
  `run_baseline.py` et `src/models/linear_probe.py`.
- Dépendances inutilisées (`wfdb`, `matplotlib`, `pyyaml`, `pandas`).
