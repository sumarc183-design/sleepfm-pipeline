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

Le contrat est contrôlé avant toute suppression de fichier. Le nettoyage des
temporaires et `--retry-skipped` concernent uniquement les sujets demandés :
les marqueurs d'autres sujets du dossier sont conservés. Les tests vérifient
le chevauchement téléchargement/embeddings et le nettoyage des deux lots en
cas d'erreur ; le gain de temps réel reste à mesurer sur un gros run.

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
sujets à calculer, pour ne jamais mélanger deux versions. Sans manifeste, un
dossier contenant déjà des caches ou rejets est refusé sans modification :
restaurer le manifeste d'origine vérifié ou repartir dans un autre dossier.
Le checkpoint passé à la commande ne suffit pas à établir cette provenance.
Les 100 caches historiques de
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

Les identifiants sont des noms uniques composés de lettres ASCII, chiffres,
tirets et underscores. Les listes vides, doublons, chemins et paramètres de
lots non positifs sont refusés avant travail. `--num-workers 0` reste permis.

**Comparer deux caches** : `python scripts/compare_embedding_caches.py ancien nouveau`
aligne les lignes par numéro d'époque et compare les stades, les embeddings
(`--atol`, 0,001 par défaut) et les temps d'origine (`--onset-atol`, 0,01 s).
Les tolérances sont absolues, finies et positives ou nulles. Sans temps complets,
le résultat est « NON VÉRIFIÉ », même avec des embeddings numériquement identiques.
Un décalage temporel connu est signalé comme différence. Les fichiers temporaires
`.tmp.npz` ne comptent pas comme sujets ; aucun cache n'est modifié.

**Sujets rejetés** : la raison est écrite dans `<sujet>.skipped` (canaux
manquants, aucun stade annoté…). Une extraction interrompue ne laisse pas de
sujet à moitié traité : un sujet n'est terminé qu'une fois son fichier de
stades écrit, en dernier.

**Bilan final du run** : `run_status.json` distingue les caches valides
(`embedded`), les rejets avec leur raison (`rejected`), les caches invalides
(`invalid`), les rejets vides ou illisibles (`invalid_rejections`), les
temporaires restants (`temporary`), les sujets manquants et les doubles résultats.
Le champ `complete` exige exactement un résultat valide par sujet demandé et
aucun temporaire pour ces sujets. Les autres sujets du dossier sont seulement
listés dans `not_requested`, sans lecture de leur contenu.

Ce contrôle relit les caches demandés un par un, avec les contraintes temporelles
du manifeste ; il ajoute donc une lecture finale des fichiers. Il ne les répare
pas et ne remplace pas le contrôle de provenance effectué avant le calcul.
Un bilan incomplet ou invalide arrête le CLI avant l'annonce de succès ou
l'entraînement des sondes. Le notebook affiche les catégories en cause avant
de proposer l'export. Les rejets ne sont valides qu'avec une raison lisible.

### Notebooks

Un notebook par usage, nommé `sleepfm_<rôle>_<données>.ipynb`, avec le même
titre en tête (« SleepFM — <rôle> <données> ») : sur Kaggle ou Colab, donner
ce titre au notebook importé pour s'y retrouver.

| Fichier | Titre | Rôle |
|---|---|---|
| `notebooks/sleepfm_embeddings_physionet2018.ipynb` | SleepFM — embeddings PhysioNet 2018 (GPU) | Calcule les embeddings SleepFM sur GPU (Kaggle ou Colab) ; sondes ensuite sur le Mac, ou ici avec `EMBED_ONLY = False` |
| `notebooks/sleepfm_raw_grid_physionet2018.ipynb` | SleepFM — prédiction sur PSG brute, grille fixe (PhysioNet 2018, GPU) | Embeddings des 100 nuits de test sur la grille fixe de 30 s, pour évaluer la prédiction sur PSG brute ; évaluation ensuite sur le Mac |

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
cache sont facultatifs pour prédire. Les `.joblib` sont des pickles : ne charger
que nos exports.

### Prédire sur une PSG brute sans annotations

Pour les fichiers PhysioNet 2018 `<sujet>.hea` et `<sujet>.mat`, sans
`-arousal.mat` :

```bash
python scripts/predict_raw_night.py tr03-0005 \
    --raw-data data/raw/physionet2018/training \
    --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt \
    --models models/physionet2018_500 --out predictions/raw --batch-size 32
```

Le script ne télécharge rien et n'entraîne aucune sonde. Il exige les 13 canaux
du montage attendu, vérifie la forme et les valeurs du signal, remet les canaux
dans l'ordre, et applique le même traitement numérique du signal que l'extracteur
annoté. Le checkpoint, la fréquence, le montage et la version du traitement sont
contrôlés contre le manifeste des sondes avant l'inférence.

Les fenêtres sont fixées à 0, 30, 60… secondes depuis le début de l'enregistrement,
sans lecture des annotations. La fin de moins de 30 secondes est retirée, sans
rembourrage. Le CSV contient les stades et probabilités ; le JSON associé indique
la grille, la durée retirée, les empreintes du brut, des modèles et du CSV, ainsi
que l'environnement. Les embeddings de cette grille ne sont pas ajoutés aux
caches d'entraînement.

**Limites :** formats EDF et autres montages non pris en charge ; nuit
rééchantillonnée en mémoire, lots de tenseurs contrôlés par `--batch-size`, CPU
par défaut et aucun worker enfant. Le BiGRU lit la nuit entière, après
enregistrement. Les fenêtres diffèrent de celles de l'entraînement annoté :
les performances publiées ne sont pas démontrées pour cette nouvelle grille.
Les tests synthétiques valident le fonctionnement, pas la qualité clinique.

**Mesurer ses performances** : le notebook `sleepfm_raw_grid_physionet2018.ipynb`
calcule sur GPU les embeddings de la grille fixe pour les 100 nuits de test du
découpage figé (`scripts/embed_raw_grid.py`). Le stade de l'expert de chaque
fenêtre est lu à part (`src/data/raw_grid_labels.py`), seulement pour noter :
une fenêtre n'est notée que si elle porte un seul stade défini ; les fenêtres à
cheval sur deux stades ou non annotées sont comptées à part. Puis, sur le Mac :

```bash
python scripts/evaluate_raw_grid.py --embeddings-dir embeddings/physionet2018_raw_grid \
    --out results/raw_grid_physionet2018
```

Les modèles figés du run à 500 sujets prédisent toute la nuit (le BiGRU voit
aussi le début non annoté). La comparaison à la grille annotée des mêmes nuits
est descriptive : les fenêtres diffèrent, il n'y a pas d'écart apparié. Les
fenêtres à cheval sur deux stades, exclues du score, sont surtout des transitions,
les plus difficiles : le score de la grille brute peut donc être optimiste.

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

## Consolider les résultats enregistrés

```bash
python scripts/consolidate_results.py \
    --out results/consolidated_physionet2018.csv \
    --markdown results/consolidated_physionet2018.md
```

Par défaut, cette commande lit uniquement les quatre JSON existants : runs de
référence à 100, 200 et 500 sujets et confirmation terminée sur les 494 sujets.
Pour d'autres fichiers, passer leurs chemins avant `--out`. Elle n'importe aucun
modèle et ne recalcule aucun score. Chaque ligne garde la source, son SHA-256,
la sonde, les effectifs disponibles, les métriques et les IC du kappa enregistrés.
Le CSV conserve la précision des valeurs ; le Markdown arrondit à trois décimales.

Une donnée historique absente reste non renseignée : ni la graine ni l'effectif
d'entraînement de la confirmation ne sont déduits d'un autre fichier. Une
confirmation incomplète, un chevauchement train/test, des effectifs incohérents
ou des métriques non finies sont refusés avant écriture. Les répétitions restent
identifiées comme telles. Le tableau ne classe pas les runs et ne démontre pas
leur comparabilité ; les différences de nuits et de prétraitement restent à
prendre en compte. MESA et OSF seront ajoutés seulement après évaluation autorisée.

## Tests

```bash
python -m pytest -q                 # unitaires + bout-en-bout
python -m pytest -q -m "not slow"    # unitaires seulement
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
