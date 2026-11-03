# Attributions

## SleepFM (`src/models/effnet.py`, parties de `src/data/extract_physionet2018.py` et de `src/models/embeddings.py`)

- Dépôt : https://github.com/rthapa84/sleepfm-codebase
- Licence : MIT (Copyright 2024 les auteurs de SleepFM)

Adapté pour la lisibilité et pour supprimer la dépendance à un clone ajouté
à `sys.path` dans third_party/. L'architecture du modèle, les conventions de
canaux et les noms de clés du checkpoint sont strictement identiques, pour
que les poids préentraînés officiels se chargent sans renommage.

## OSF (`src/models/osf_vit1d.py`)

- Dépôt : https://github.com/yang-ai-lab/OSF-Open-Sleep-FM
- Licence : MIT

Réimplémentation minimale de `osf/backbone/vit1d_cls.py::ViT` (ViT 1D avec
token CLS), suffisante pour charger le `osf_backbone.pth` publié et produire
des embeddings, sans reprendre tout le code d'entraînement. Voir
`docs/OSF_COMPARISON.md` pour les différences de canaux et de schéma de
labels à traiter avant de comparer directement SleepFM et OSF.
