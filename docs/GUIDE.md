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
python run_baseline.py --physionet-subjects 200 --embeddings-dir embeddings/physionet2018_v2 \
    --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt \
    --split-file configs/split_physionet2018.json
```

Chaque lot de sujets (`--chunk-size`, 8 par défaut) est téléchargé, extrait et
passé dans SleepFM, puis le brut et les époques sont supprimés : seul un `.npz`
d'embeddings par sujet (~5 Mo, contre ~0,85 Go de brut et d'époques) reste dans
`embeddings/`. Un sujet déjà présent n'est pas recalculé : relancer reprend là
où on s'était arrêté. Chaque `.npz` est écrit sous un nom temporaire puis
renommé, et un cache illisible (écriture interrompue) est recalculé au
démarrage : interrompre un run, même en pleine écriture, ne casse pas le
suivant. Même split, mêmes époques et mêmes labels que le mode
classique (vérifié par `tests/test_pipeline_smoke.py`).

Chaque `.npz` contient `X` (embeddings), `y` (stades), `epoch_idx` (numéro
d'époque, pour remettre la nuit dans l'ordre chronologique : les fichiers sont
triés par nom, `_10` avant `_2`) et `onset_s` (début de l'époque, en secondes
depuis le début de l'enregistrement d'origine : un écart de plus de 30 s
signale des époques retirées en cours de nuit). Pour un cache antérieur à
`epoch_idx`, l'ordre est reconstitué depuis ce tri.

**Découpage figé** (`--split-file`) : `configs/split_physionet2018.json` liste
les 400 sujets d'entraînement et les 100 de test du run de référence (les rôles
des runs à 100 et 200 sujets sont conservés). Un sujet
listé garde toujours son rôle ; les nouveaux sujets sont répartis (même
fraction, même graine) et ajoutés au fichier. Sans ce fichier, passer à 200
sujets refait tout le tirage et change les nuits de test.

**Provenance du cache** : le dossier garde un `manifest.json` (empreinte
SHA-256 du checkpoint, version du prétraitement, canaux, fréquence). Un autre
checkpoint est refusé ; un autre prétraitement aussi, dès qu'il reste des
sujets à calculer, pour ne jamais mélanger deux versions. Les 100 caches de
`embeddings/physionet2018` datent d'avant le manifeste : ils sont déclarés en
version 1 (début utile pris sur la colonne « wake »). La version 2 prend la fin
de la période « undefined », colonne désignée par son nom, comme le code
officiel. Sur les 100 mêmes nuits, 91 sont identiques et les 9 autres récupèrent
480 époques de début de nuit. Le run de référence utilise uniquement
`embeddings/physionet2018_v2` (500 sujets en version 2).

**Contrat du cache** : checkpoint, prétraitement, canaux dans le même ordre et
fréquence. Il est comparé en entier pour réunir des caches
(`src/data/cache_contract.py`, utilisé par l'import Kaggle) et pour prédire avec
des sondes exportées. Chaque fichier est aussi vérifié : formes, valeurs finies,
indices d'époque entiers et uniques, temps tous finis et croissants ou tous
absents (cache historique). Un temps manquant isolé est refusé.

**Sujets rejetés** : la raison est écrite dans `<sujet>.skipped` (canaux
manquants, aucun stade annoté…). Une extraction interrompue ne laisse pas de
sujet à moitié traité : un sujet n'est terminé qu'une fois son fichier de
stades écrit, en dernier.

### Notebooks

Un notebook par usage, nommé `sleepfm_<rôle>_<données>.ipynb`, avec le même
titre en tête (« SleepFM — <rôle> <données> ») : sur Kaggle ou Colab, donner
ce titre au notebook importé pour s'y retrouver.

| Fichier | Titre | Rôle |
|---|---|---|
| `notebooks/sleepfm_embeddings_physionet2018.ipynb` | SleepFM — embeddings PhysioNet 2018 (GPU) | Calcule les embeddings SleepFM sur GPU (Kaggle ou Colab) ; sondes ensuite sur le Mac, ou ici avec `EMBED_ONLY = False` |

### Google Colab (GPU gratuit)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumarc183-design/sleepfm-pipeline/blob/main/notebooks/sleepfm_embeddings_physionet2018.ipynb)

Sélectionner `GPU T4` dans le type d'exécution, puis `Tout exécuter`. Le
notebook utilise le mode par lots (500 sujets par défaut) avec les embeddings
sur Google Drive : une déconnexion ne fait rien perdre. Sorties :
`results_physionet2018_<N>.txt` (lisible) et `.json` (métriques, sujets de
train et de test, réglages et convergence, pour comparer deux runs).

### Kaggle (autre GPU gratuit)

[![Ouvrir dans Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/sumarc183-design/sleepfm-pipeline/blob/main/notebooks/sleepfm_embeddings_physionet2018.ipynb)

Même notebook, qui détecte Kaggle tout seul. Utile quand le quota GPU de Colab
est épuisé : Kaggle donne environ 30 h de GPU par semaine.
`Settings > Accelerator > GPU T4 x2` (pas P100, trop ancien pour PyTorch
récent) et `Settings > Internet > On`. Pour un long run (200 sujets,
~2 h 30), `Save Version > Save & Run All (Commit)` : le calcul continue
navigateur fermé (jusqu'à 12 h) et les sorties restent dans l'onglet `Output`
de la version. `Run All` en session interactive s'arrête si la session expire.

Le notebook calcule par défaut 500 sujets (100 nuits de test, comme
l'article ; ~1 h 15 de GPU par tranche de 100 sujets à calculer) dans
`embeddings/physionet2018_v2`
(prétraitement version 2, découpage figé), en mode **embeddings seulement**
(`EMBED_ONLY = True`, option `--embed-only`) : le GPU ne sert qu'à SleepFM,
et les sondes, qui n'en ont pas besoin, tournent ensuite sur le Mac. Cela
économise ~20 à 25 min de GPU par run. Le téléchargement des nuits PhysioNet
compte aussi dans le quota, puisque la session tourne pendant ce temps : ne
recalculer que les sujets manquants, en important les embeddings déjà faits. Kaggle n'a pas accès à Google
Drive : à la fin, les embeddings sont zippés dans `Output`
(`embeddings_physionet2018_v2.zip`, ~5 Mo par sujet), avec le découpage
étendu, avec les résultats et les
prédictions. Les dézipper dans `embeddings/physionet2018_v2/` permet de
relancer les sondes sur le Mac sans GPU. Pour ne pas recalculer sur Kaggle,
publier ce dossier comme Dataset Kaggle **privé** et l'ajouter au notebook
(`Add Input`) : il n'est copié que si son `manifest.json` correspond au
prétraitement actuel.

## Sondes et sélection de modèle

Toutes les sondes s'entraînent sur les mêmes embeddings figés et le même split
**par sujet** (20 % de test par défaut, `--test-fraction`), jamais par époque : les époques d'une
même nuit se ressemblent, les séparer gonflerait les scores.

| Sonde (`--probes`) | Levier testé |
|---|---|
| `logistic` | **Baseline** : protocole officiel SleepFM (`class_weight="balanced"`), seule comparable à la publication |
| `logistic_unweighted` | Même modèle sans pondération des classes (baseline des premiers résultats) |
| `xgboost` | Frontière non linéaire par arbres (GPU si disponible) |
| `mlp` | Frontière non linéaire par réseau (1 couche cachée de 256), plus rapide |
| `sequence` | Contexte temporel : BiGRU sur la suite des embeddings de la nuit (`src/models/sequence_probe.py`) |

Métriques pour chacune : accuracy, kappa de Cohen, F1 macro, ROC-AUC et
AUPRC un-contre-tous (macro et par stade), F1 par stade, le **gap** train − test (un gap élevé signale du
surapprentissage), l'**intervalle de confiance à 95 %** du kappa et du F1
macro (bootstrap sur les sujets de test : on tire des nuits entières, pas des
époques, qui ne sont pas indépendantes) et la **matrice de confusion** de la
baseline. Hyperparamètres fixes, jamais réglés sur le test.

**Comparaison appariée** : pour chaque sonde, l'écart de κ et de F1 macro à
la baseline, avec son IC à 95 %, par bootstrap sur les **mêmes** nuits de
test pour les deux sondes. Deux IC séparés se chevauchent presque toujours ;
l'écart apparié retire la variabilité commune (nuits faciles ou
difficiles). Un écart dont l'IC exclut 0 est significatif.

**Baseline officielle et convergence** : le dépôt officiel entraîne
`LogisticRegression(class_weight="balanced")` (`sleepfm/utils.py::train_model`).
Son option `--max_iter` n'est pas transmise au modèle, qui garde 100
itérations sans garantie de convergence. Ici, `--max-iter` vaut 1 000 et la
sortie affiche le nombre d'itérations utilisées, ou un avertissement explicite
si le modèle n'a pas convergé.

**Sondes exportées** (`--save-models DIR`, avec `--results-json`) : chaque
sonde entraînée est écrite dans `DIR/<sonde>.joblib`, avec un
`manifest.json` (contrat d'entrée et dimension des embeddings ; classes ;
paramètres effectifs, nuits de validation et meilleur passage de chaque sonde ;
commit et versions). `scripts/predict_night.py` les recharge et écrit
l'hypnogramme prédit d'une nuit en CSV (stade et probabilités par époque). Il
refuse tout cache au contrat absent, incomplet ou différent, sauf
`--allow-unverified-provenance`, réservé aux caches historiques. Les stades du
cache sont facultatifs pour prédire, mais une PSG brute jamais annotée n'est pas
encore prise en charge : l'extraction choisit ses fenêtres à partir des
annotations. Les `.joblib` sont des pickles : ne charger que nos exports.

**Réutiliser les embeddings sur Kaggle** : publier le zip
`embeddings_physionet2018_v2.zip` comme Dataset Kaggle privé, puis l'ajouter
au notebook (`Add Input`) : seuls les sujets manquants sont recalculés. Pour
relancer les sondes sur les mêmes sujets, Kaggle n'est plus utile : le Mac
suffit.

**Prédictions sauvegardées** : avec `--results-json run.json`, le script écrit
aussi `run.predictions.npz` : pour chaque époque de test, le sujet, le numéro
d'époque, le vrai stade, puis la prédiction et les probabilités de chaque
sonde. IC, comparaisons appariées et analyse d'erreurs se recalculent sans
réentraîner.

**BiGRU : longueurs réelles et ablation du contexte**. L'entraînement se fait
sur des fenêtres d'1 h (120 époques), complétées par des zéros dans un lot ;
le GRU reçoit la longueur réelle de chaque fenêtre (séquences compactées),
pour que le rembourrage n'influence pas les vraies époques. La prédiction lit
la nuit entière. `scripts/ablation_sequence.py` mesure d'où vient le gain :
contexte à la prédiction (nuit, 1 h, 5 min, 1 époque) et ordre des époques
mélangé dans chaque nuit. `scripts/analyse_errors.py` relit un
`.predictions.npz` et compare deux sondes : matrices de confusion, kappa par
nuit, précision selon la distance au changement de stade, et détail de N1.
Le BiGRU voit les époques futures : il convient à
une analyse après l'enregistrement, pas au temps réel.

**Arrêt précoce sans fuite** : le MLP et le BiGRU s'arrêtent sur ~10 % des
**nuits** du train mises à part. L'arrêt précoce intégré de scikit-learn
tirerait des époques au hasard, issues des mêmes nuits que l'entraînement.

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

**Mac Intel avec Anaconda** : XGBoost plante ou se bloque une fois que PyTorch a
calculé dans le même processus (deux runtimes OpenMP), même avec `n_jobs=1`.
`run_baseline.py` le retire donc des sondes sur Mac, avec un avertissement, et
ses tests y sont sautés. Lancer XGBoost sur Linux (Colab, Kaggle, CI).

**Environnement restreint** : si le DataLoader ne peut pas lancer de processus
(`torch_shm_manager: Operation not permitted`), ajouter `--num-workers 0`. Par
défaut, le nombre de workers des embeddings suit `--num-threads`.

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

## Points d'attention connus

- **Double rééchantillonnage à l'extraction** : `raw.resample(256)` puis
  `scipy.signal.resample` vers `--target_sample_rate`. Sans effet avec la
  valeur par défaut (256) ; une autre valeur donnerait un double
  rééchantillonnage. Hérité du prétraitement officiel, gardé tel quel pour
  rester comparable.
- **Reprise de l'extraction en mode classique** : un sujet est considéré comme
  traité si ses fichiers `Y/` et `T/` se relisent et que ses fichiers d'époques
  existent. Le contenu de chaque fichier `X/` n'est pas relu : après une
  corruption, supprimer le dossier du sujet. Le mode par lots n'est pas concerné
  (dossiers temporaires).
- **Sous-ensemble de sujets** : `--physionet-subjects N` prend les N premiers
  sujets par identifiant, pas un tirage aléatoire.

## Code adapté du dépôt officiel

`src/models/effnet.py`, `src/data/extract_physionet2018.py` et
`src/models/embeddings.py` sont adaptés du dépôt officiel SleepFM (voir
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)) pour rester strictement
compatibles avec les checkpoints publiés : mêmes noms de clés, même montage
de canaux, même architecture.

## Confirmation sur les sujets réservés

À lancer une seule fois, après validation de [`PROTOCOLE_CONFIRMATION.md`](PROTOCOLE_CONFIRMATION.md).

1. Embeddings seuls, sans sonde ni extension du découpage (Kaggle : `CONFIRMATION = True` dans le
   notebook, ou) :
   `python run_baseline.py --embed-only --subjects-file configs/reserved_physionet2018.json --embeddings-dir embeddings/physionet2018_reserved_v2 --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt`
2. Évaluation des modèles figés, sans entraînement :
   `python scripts/confirm_frozen_models.py --embeddings-dir embeddings/physionet2018_reserved_v2 --out results/confirmation_physionet2018`

La commande vérifie d'abord les empreintes (modèles, liste, découpage), le contrat et la couverture
exacte des sujets, et ne calcule aucun score en cas d'écart. `--rehearsal <sujets déjà ouverts>` la teste
sans toucher aux sujets réservés.
