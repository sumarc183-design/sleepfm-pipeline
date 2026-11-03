# SleepFM Pipeline

Reproduction et évaluation de [SleepFM](https://github.com/rthapa84/sleepfm-codebase) (modèle fondamental multimodal pour le sommeil) sur données ouvertes, avec comparaison à [OSF](https://github.com/yang-ai-lab/OSF-Open-Sleep-FM) / SleepBench.

## Résultats

| Date | Données | Checkpoint | Accuracy | Cohen's κ | Macro F1 |
|---|---|---|---|---|---|
| 2026-11-02 | PhysioNet 2018, 20 sujets (15 train / 5 test, split par sujet, 4 295 époques de test) | `best.pt` officiel | 0,703 | **0,584** | 0,650 |

Sonde logistique sur embeddings figés (combinés : respiratoire + stades + ECG),
5 classes. Lancé sur Colab GPU T4 via `notebooks/colab_baseline_physionet2018.ipynb`,
sortie brute dans `results/results_physionet2018.txt`.

Référence interne, pas une reproduction du chiffre publié : le papier SleepFM
rapporte la stadification sur MESA/SHHS, pas sur PhysioNet 2018. Avec 5 sujets
de test, le κ peut varier d'environ ±0,05 selon le tirage ; à consolider avec
plus de sujets, puis comparer au chiffre publié sur MESA.

## Objectif phase 1

Ne pas ré-entraîner un modèle fondamental. Reproduire la baseline publiée sur une tâche mesurable — classification des stades du sommeil (5 classes : Wake, N1, N2, N3, REM) — avant d'investir dans du développement propriétaire.

## Approche

1. **Reproduction** : pipeline officiel SleepFM (poids préentraînés) → génération d'embeddings → sonde linéaire → comparaison au chiffre publié.
2. **Comparaison** : mêmes données, mêmes métriques, contre OSF/SleepBench.
3. **Différenciation** : une fois la baseline établie, cibler un angle où l'on peut mesurablement faire mieux (ex. robustesse pédiatrique via CHAT, détection d'événements respiratoires).

## Données

| Base | Statut | Usage |
|---|---|---|
| PhysioNet Challenge 2018 | Public, téléchargement immédiat (994 sujets d'entraînement annotés) | Validation du pipeline (phase 1) |
| MESA (NSRR) | Accès sur demande (DUA) | Reproduction de la démo officielle SleepFM |
| SHHS (NSRR) | Accès sur demande (DUA) | Évaluation à plus grande échelle (phase 2) |
| CHAT (NSRR) | Accès sur demande (DUA) | Pédiatrie — piste de différenciation (phase 2) |

Voir `docs/DATA_ACCESS.md` pour la procédure de demande d'accès NSRR.

## Structure

```
run_baseline.py point d'entrée unique : extraction -> embeddings -> sonde -> évaluation
configs/        conventions de canaux/labels de référence (non lues par les scripts)
scripts/        téléchargement de données, setup des dépôts tiers, générateur synthétique
src/data/       chargement et prétraitement des PSG
src/models/     wrapper SleepFM (embeddings), sonde linéaire, backbone OSF
src/eval/       métriques (accuracy, Cohen's kappa, F1 par classe)
notebooks/      notebook Google Colab (GPU gratuit)
results/        sorties brutes des runs réels
docs/           roadmap, accès aux données NSRR, comparaison OSF
```

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup_sleepfm.sh   # clone le code officiel SleepFM dans third_party/ pour récupérer best.pt
python scripts/download_physionet2018.py --out data/raw/physionet2018/training --limit 20
python run_baseline.py --raw-data data/raw/physionet2018/training \
    --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt
```

`run_baseline.py` fait le split par sujet (25 % en test), l'extraction, les
embeddings et la sonde en une commande. Sans GPU local, passer par le
notebook Colab (voir plus bas).

## Tester sans données réelles (smoke test)

`scripts/generate_synthetic_physionet2018.py` génère un jeu de données au
format exact PhysioNet 2018 (bruit aléatoire, pas de vraie physiologie) pour
faire tourner tout le pipeline et vérifier que la plomberie fonctionne, en
attendant l'accès aux vraies données :

```bash
pip install -r requirements-dev.txt
pytest tests/ -v                 # tests unitaires + bout-en-bout (~1 min)
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

Ce test vérifie le format des époques extraites (845 époques de 30 s pour
`tr03-0005`, canaux identiques à `ALL_CHANNELS`) et fait tourner embeddings
et sonde avec des poids aléatoires : il valide la plomberie, pas une
performance (accuracy ~0,20, niveau du hasard sur 5 classes). Les vrais
chiffres, avec le checkpoint officiel, sont dans "Résultats". Sauté
automatiquement si `data/processed/train` est absent (CI compatible).

## Lancer sur GPU gratuit (Google Colab)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumarc183-design/sleepfm-pipeline/blob/main/notebooks/colab_baseline_physionet2018.ipynb)

`notebooks/colab_baseline_physionet2018.ipynb` récupère `best.pt` et lance
`run_baseline.py` en mode par lots sur N sujets PhysioNet 2018 (100 par
défaut), avec les embeddings sur GPU (`--device cuda`, choisi automatiquement
si disponible). Sélectionner `GPU T4` dans le type d'exécution avant de lancer.

Mode par lots (`--physionet-subjects N --embeddings-dir DIR`) : chaque lot de
sujets est téléchargé, extrait et passé dans SleepFM, puis le brut et les
époques sont supprimés ; seul un `.npz` d'embeddings par sujet (~5 Mo, contre
~0,85 Go de brut + époques) reste dans `DIR`. Un sujet déjà présent n'est pas
recalculé : avec `DIR` sur Google Drive, une déconnexion Colab ne fait rien
perdre. Même split par sujet, mêmes époques et mêmes labels que le mode classique (vérifié
par `tests/test_pipeline_smoke.py`).

## Checkpoint officiel

Le checkpoint pré-entraîné `best.pt` (~84 Mo) est hébergé directement dans le
dépôt officiel, récupérable via `bash scripts/setup_sleepfm.sh` (clone
`third_party/sleepfm/`, checkpoint dans `third_party/sleepfm/sleepfm/checkpoint/best.pt`).

Validé ici : chargement réel du `best.pt` publié, génération d'embeddings,
`run_baseline.py --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt`
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
évaluation) tourne de bout en bout avec le vrai checkpoint officiel, et la
suite de tests passe en CI. Premier vrai chiffre obtenu sur 20 sujets PhysioNet 2018
avec le checkpoint officiel (voir "Résultats" en haut de ce fichier).

## CI

`.github/workflows/ci.yml` lance la suite de tests à chaque push/PR.

## Roadmap

Voir `docs/ROADMAP.md`.
