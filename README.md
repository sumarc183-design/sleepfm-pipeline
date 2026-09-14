# SleepFM Pipeline

Reproduction et évaluation de [SleepFM](https://github.com/rthapa84/sleepfm-codebase) (modèle fondamental multimodal pour le sommeil) sur données ouvertes, avec comparaison à [OSF](https://github.com/bdsp-core/OSF) / SleepBench.

## Objectif phase 1

Ne pas ré-entraîner un modèle fondamental. Reproduire la baseline publiée sur une tâche mesurable — classification des stades du sommeil (5 classes : Wake, N1, N2, N3, REM) — avant d'investir dans du développement propriétaire.

## Approche

1. **Reproduction** : pipeline officiel SleepFM (poids préentraînés) → génération d'embeddings → sonde linéaire → comparaison au chiffre publié.
2. **Comparaison** : mêmes données, mêmes métriques, contre OSF/SleepBench.
3. **Différenciation** : une fois la baseline établie, cibler un angle où l'on peut mesurablement faire mieux (ex. robustesse pédiatrique via CHAT, détection d'événements respiratoires).

## Données

| Base | Statut | Usage |
|---|---|---|
| PhysioNet Challenge 2018 | Public, téléchargement immédiat | Validation du pipeline (phase 1) |
| MESA (NSRR) | Accès sur demande (DUA) | Reproduction de la démo officielle SleepFM |
| SHHS (NSRR) | Accès sur demande (DUA) | Évaluation à plus grande échelle (phase 2) |
| CHAT (NSRR) | Accès sur demande (DUA) | Pédiatrie — piste de différenciation (phase 2) |

Voir `docs/DATA_ACCESS.md` pour la procédure de demande d'accès NSRR.

## Structure

```
configs/        configuration (chemins, hyperparamètres)
scripts/        téléchargement de données, setup des dépôts tiers
src/data/       chargement et prétraitement des PSG
src/models/     wrapper SleepFM (embeddings), sonde linéaire
src/eval/       métriques (accuracy, Cohen's kappa, F1 par classe)
docs/           roadmap et notes d'accès aux données
```

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_physionet2018.py --out data/raw/physionet2018
bash scripts/setup_sleepfm.sh   # clone le code officiel SleepFM dans third_party/
```

## Roadmap

Voir `docs/ROADMAP.md`.
