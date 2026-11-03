# Guide d'utilisation

Détail de tout ce que le [README](../README.md) résume : installation, modes
d'exécution, sondes, tests et points d'attention techniques.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # dépendances + pytest + ruff
pre-commit install                    # optionnel : lint et format à chaque commit
bash scripts/setup_sleepfm.sh         # clone le code officiel SleepFM dans third_party/ (best.pt)
```

Python 3.10 ou plus. `requirements.txt` borne chaque dépendance à la version
majeure validée (Mac Intel et Colab).

## Lancer la baseline

`run_baseline.py` est le point d'entrée unique : split par sujet, extraction
des époques, embeddings SleepFM figés, sondes, évaluation.

### Mode classique (quelques dizaines de sujets)

```bash
python scripts/download_physionet2018.py --out data/raw/physionet2018/training --limit 20
python run_baseline.py --raw-data data/raw/physionet2018/training \
    --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt \
    --results-json results/run.json
```

Sans `--raw-data` : données synthétiques. Sans `--checkpoint` : poids
aléatoires. Dans les deux cas, la sortie le signale (plomberie uniquement).

### Mode par lots (100 sujets et plus)

```bash
python run_baseline.py --physionet-subjects 200 --embeddings-dir embeddings/ \
    --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt
```

Chaque lot de sujets (`--chunk-size`, 8 par défaut) est téléchargé, extrait et
passé dans SleepFM, puis le brut et les époques sont supprimés : seul un `.npz`
d'embeddings par sujet (~5 Mo, contre ~0,85 Go de brut et d'époques) reste dans
`embeddings/`. Un sujet déjà présent n'est pas recalculé : relancer reprend là
où on s'était arrêté. Même split, mêmes époques et mêmes labels que le mode
classique (vérifié par `tests/test_pipeline_smoke.py`).

Chaque `.npz` contient `X` (embeddings), `y` (stades) et `epoch_idx` (numéro
d'époque, pour remettre la nuit dans l'ordre chronologique : les fichiers sont
triés par nom, `_10` avant `_2`). Pour un cache antérieur à `epoch_idx`, l'ordre
est reconstitué depuis ce tri.

### Google Colab (GPU gratuit)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumarc183-design/sleepfm-pipeline/blob/main/notebooks/colab_baseline_physionet2018.ipynb)

Sélectionner `GPU T4` dans le type d'exécution, puis `Tout exécuter`. Le
notebook utilise le mode par lots (100 sujets par défaut) avec les embeddings
sur Google Drive : une déconnexion ne fait rien perdre. Sorties :
`results_physionet2018_<N>.txt` (lisible) et `.json` (métriques structurées).

## Sondes et sélection de modèle

Toutes les sondes s'entraînent sur les mêmes embeddings figés et le même split
**80/20 par sujet** (`--test-fraction`), jamais par époque : les époques d'une
même nuit se ressemblent, les séparer gonflerait les scores.

| Sonde (`--probes`) | Levier testé |
|---|---|
| `logistic` | **Baseline** : protocole officiel SleepFM, seule comparable à la publication |
| `logistic_balanced` | Déséquilibre des classes (N1 rare) : classes pondérées |
| `xgboost` | Frontière non linéaire par arbres (GPU si disponible) |
| `mlp` | Frontière non linéaire par réseau (1 couche cachée de 256), plus rapide |
| `sequence` | Contexte temporel : BiGRU sur la suite des embeddings de la nuit (`src/models/sequence_probe.py`) |

Métriques pour chacune : accuracy, kappa de Cohen, F1 macro, ROC-AUC
un-contre-tous, F1 par stade, et le **gap** train − test (un gap élevé signale
du surapprentissage). Hyperparamètres fixes, jamais réglés sur le test (arrêt
précoce du MLP et du BiGRU sur une partie du train).

### MLP ou XGBoost

Une seule sonde non linéaire est gardée. Le choix se fait **sur le train
uniquement**, par validation croisée à 5 plis groupée par sujet (`--cv-folds`,
mêmes plis pour les deux sondes) ; le test reste à part jusqu'au bout et la
sonde écartée n'y est jamais évaluée :

1. une sonde dont le gap F1 moyen (apprentissage − validation) dépasse 0,15
   (`--max-gap`) est éliminée quel que soit son score : elle apprend par cœur ;
2. si les deux restent, XGBoost seulement s'il bat le MLP de plus de 0,02 en
   F1 macro de validation en moyenne (`--min-gain`) **et** sur au moins 4 plis
   sur 5 (un gain qui ne tient que sur quelques plis dépend du tirage) ;
   sinon le MLP, plus rapide ;
3. si aucune ne reste, on s'en tient à la régression logistique.

La sortie affiche d'abord cette sélection (F1 de validation par pli), puis
l'évaluation finale sur le test des sondes retenues. Random Forest, testé, a
été retiré (plus lent que XGBoost et moins adapté aux embeddings denses).

## Tests

```bash
pytest tests/ -v                 # unitaires + bout-en-bout (~1 min, quelques minutes sur CPU lent)
pytest tests/ -m "not slow"      # unitaires seulement
ruff check . && ruff format --check .
```

Les tests de bout en bout tournent sur des données synthétiques au format
exact PhysioNet 2018 (`scripts/generate_synthetic_physionet2018.py` : bruit,
pas de vraie physiologie). La CI lance lint puis tests à chaque push.

**Sur vraies données** :

```bash
python src/data/extract_physionet2018.py --data_path data/raw/physionet2018/training --save_path data/processed/train
pytest tests/test_real_data.py -v -s
```

Vérifie le format des époques extraites (845 époques de 30 s pour
`tr03-0005`, canaux identiques à `ALL_CHANNELS`) et fait tourner embeddings et
sonde avec des poids aléatoires : plomberie, pas performance. Sauté si
`data/processed/train` est absent (CI).

**Mac Intel avec Anaconda** : XGBoost multithread plante (segfault) une fois
PyTorch chargé dans le même processus (deux runtimes OpenMP). Préfixer les
commandes par `OMP_NUM_THREADS=1`. Sans effet sur Colab/Linux.

**Réseau d'entreprise** : si le téléchargement Python de PhysioNet est bloqué
par un proxy, chaque sujet ne demande que 3 fichiers, téléchargeables au
navigateur (voir la docstring de `scripts/download_physionet2018.py`).

## Checkpoint officiel

`best.pt` (~84 Mo) est versionné dans le dépôt officiel SleepFM, récupéré par
`bash scripts/setup_sleepfm.sh` dans `third_party/sleepfm/sleepfm/checkpoint/best.pt`.

**Piège découvert en le chargeant** : le `best.pt` publié utilise les clés
`resp_state_dict` / `sleep_state_dict` / `ekg_state_dict`, alors que le propre
script d'inférence du dépôt officiel (`3_generate_embed_pretraining.py`)
attend `respiratory_state_dict` / `sleep_stages_state_dict` / `ekg_state_dict`.
Incohérence du dépôt amont (vérifié dans `2_pretrain.py` : les noms longs sont
ceux d'un réentraînement complet), pas une erreur introduite ici.
`src/models/embeddings.py` accepte les deux formes.

## Code adapté du dépôt officiel

`src/models/effnet.py`, `src/data/extract_physionet2018.py` et
`src/models/embeddings.py` sont adaptés du dépôt officiel SleepFM (voir
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)) pour rester strictement
compatibles avec les checkpoints publiés : mêmes noms de clés, même montage
de canaux, même architecture.
