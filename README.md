# SleepFM Pipeline

Reproduction et évaluation de [SleepFM](https://github.com/rthapa84/sleepfm-codebase) (modèle fondamental multimodal pour le sommeil) sur données ouvertes, avec comparaison à [OSF](https://github.com/yang-ai-lab/OSF-Open-Sleep-FM) / SleepBench.

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
python src/data/extract_physionet2018.py --data_path data/raw/physionet2018 --save_path data/processed/train
bash scripts/setup_sleepfm.sh   # clone le code officiel SleepFM dans third_party/ pour récupérer best.pt
python scripts/run_embedding_and_probe.py --processed-train data/processed/train --processed-test data/processed/test
```

## Tester sans données réelles (smoke test)

`scripts/generate_synthetic_physionet2018.py` génère un jeu de données au
format exact PhysioNet 2018 (bruit aléatoire, pas de vraie physiologie) pour
faire tourner tout le pipeline et vérifier que la plomberie fonctionne, en
attendant l'accès aux vraies données :

```bash
pip install -r requirements-dev.txt
pytest tests/ -v                 # tests unitaires + bout-en-bout (~15s)
pytest tests/ -m "not slow"      # unitaires seulement, rapide
```

Le test `tests/test_pipeline_smoke.py::test_full_pipeline_end_to_end` fait
tourner génération -> extraction -> embeddings -> sonde -> évaluation en une
passe. À REMPLACER par les vraies données PhysioNet 2018/MESA dès que
possible : ces résultats synthétiques ne sont pas interprétables
médicalement (kappa proche de 0 attendu, c'est du bruit).

## Tester sur les vraies données

Une fois quelques enregistrements PhysioNet 2018 téléchargés (voir
"Démarrage" ci-dessus -- même à la main via le navigateur si le
téléchargement Python est bloqué par un proxy d'entreprise, voir la
docstring de `scripts/download_physionet2018.py`) et extraits dans
`data/processed/train/` :

```bash
pytest tests/test_real_data.py -v -s
```

Validé en conditions réelles : extraction sur 1 enregistrement PhysioNet
2018 officiel (845 époques de 30s, format confirmé identique à
`ALL_CHANNELS`), embeddings générés, sonde ajustée -- accuracy ~0.20
(niveau du hasard sur 5 classes), cohérent avec un backbone à poids
aléatoires (pas de checkpoint SleepFM officiel téléchargé). Avec un seul
sujet, pas de split train/test significatif : ce test valide la plomberie,
pas une performance de généralisation. Sauté automatiquement si
`data/processed/train` est absent (CI compatible).

## Checkpoint officiel

Le checkpoint pré-entraîné `best.pt` (~84 Mo) est hébergé directement dans le
dépôt officiel, récupérable via `bash scripts/setup_sleepfm.sh` (clone
`third_party/sleepfm/`, checkpoint dans `third_party/sleepfm/checkpoint/best.pt`).

Validé ici : chargement réel du `best.pt` publié, génération d'embeddings,
`run_baseline.py --checkpoint third_party/sleepfm/checkpoint/best.pt`
fonctionne de bout en bout.

**Piège découvert en le chargeant** : le `best.pt` publié utilise les clés
`resp_state_dict`/`sleep_state_dict`/`ekg_state_dict`, alors que le propre
script d'inférence du dépôt officiel (`3_generate_embed_pretraining.py`)
attend `respiratory_state_dict`/`sleep_stages_state_dict`/`ekg_state_dict` —
une incohérence dans le dépôt amont, pas une erreur introduite ici (vérifié
en lisant `2_pretrain.py` : les noms longs sont bien ceux produits par un
réentraînement complet, mais le fichier `best.pt` livré tel quel utilise les
noms courts). `src/models/embeddings.py` accepte les deux formes.

## État du code

`src/models/effnet.py`, `src/data/extract_physionet2018.py` et
`src/models/embeddings.py` sont adaptés du dépôt officiel SleepFM (voir
`THIRD_PARTY_NOTICES.md`) pour rester strictement compatibles avec les
checkpoints publiés (mêmes noms de clés -- avec gestion des deux variantes
ci-dessus --, même montage de canaux, même architecture). Le pipeline
complet (génération synthétique -> extraction -> embeddings -> sonde ->
évaluation) tourne de bout en bout avec le vrai checkpoint officiel, 11/11
tests passent. Le téléchargement réel de PhysioNet 2018 n'a pas pu être
validé depuis cet environnement de développement (physionet.org hors de la
liste d'hôtes autorisés du sandbox) — à lancer depuis votre poste ou une
instance cloud classique. Le prochain vrai chiffre nécessite plus de sujets
PhysioNet 2018 téléchargés (`--raw-data`) combinés à ce checkpoint réel.

## CI

`.github/workflows/ci.yml` lance la suite de tests à chaque push/PR.

## Roadmap

Voir `docs/ROADMAP.md`.
