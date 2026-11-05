# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté
- **Spectre de l'EEG par groupe d'époques** (`scripts/eeg_spectra_by_group.py`) :
  bandes de C3, alpha occipital et tonus du menton, par couple stade de
  l'expert / stade prédit, pour comprendre un désaccord.
- **Comparaison anciennes / nouvelles nuits de test**
  (`scripts/compare_test_nights.py`) : quand le découpage est étendu, κ sur
  les anciennes nuits de test, les nouvelles et l'ensemble, et nuits au κ le
  plus bas.
- **Découpage figé** (`--split-file`, `configs/split_physionet2018.json`) :
  un sujet garde son rôle quand on en ajoute ; les 80 / 20 sujets du run à
  100 sujets y sont enregistrés, étendus à 160 / 40 pour 200 sujets.
- **Manifeste du cache d'embeddings** (`manifest.json`) : empreinte du
  checkpoint, version du prétraitement, canaux, fréquence. Un cache d'un
  autre checkpoint est refusé, un autre prétraitement n'est jamais mélangé.
  Enregistré dans le JSON des résultats.
- **Temps d'origine de chaque époque** (`onset_s`) dans l'extraction et le
  cache ; **raison des rejets** enregistrée.
- **BiGRU pondéré** : option `class_weight="balanced"` de la sonde
  séquentielle et `scripts/compare_weighted_sequence.py`.
- **Notebook renommé** `notebooks/sleepfm_embeddings_physionet2018.ipynb`
  (« SleepFM — embeddings PhysioNet 2018 (GPU) ») ; règle de nommage des
  notebooks dans le GUIDE.
- **Export des sondes** (`--save-models`, `src/models/probe_export.py`) :
  sondes rechargeables avec leur manifeste (contrat d'entrée, classes,
  hyperparamètres, commit, versions) ; `scripts/predict_night.py` prédit
  l'hypnogramme d'une nuit en CSV, après contrôle du checkpoint et du
  prétraitement.
- **Comparaison de deux caches** (`scripts/compare_embedding_caches.py`) :
  mêmes époques, mêmes stades, embeddings identiques sur les sujets communs
  (pour vérifier le prétraitement version 2 sur les 100 premiers sujets).
- **CI** : tests aussi en Python 3.10, version minimale annoncée.
- **Examen des signaux bruts** (`scripts/inspect_night_signals.py`) : qualité
  des canaux et physiologie par stade (tonus du menton, mouvements oculaires,
  alpha). Deux nuits ratées : REM pris pour de l'éveil, EMG du menton quasi
  plat dans l'une, REM atypique dans l'autre.
- **`--embed-only`** (mode par lots) : calcule seulement les embeddings. Le
  notebook l'active par défaut : le GPU ne sert qu'à SleepFM, les sondes
  tournent ensuite sur le Mac (~20 à 25 min de GPU économisées par run).
- **Notebook** : 500 sujets par défaut (100 nuits de test, comme
  l'article) ; un Dataset Kaggle qui garde le zip des embeddings tel quel est
  décompressé avant l'import ; le découpage étendu est copié dans `Output`.
- **Notebook** (2026-11-04) : 200 sujets par défaut, dans `physionet2018_v2` avec le
  découpage figé ; sur Kaggle, n'importe que des embeddings au même
  prétraitement, et place prédictions et zip des embeddings dans `Output`.
- **Prédictions du test sauvegardées** (`<json>.predictions.npz`) : sujet,
  numéro d'époque, vrai stade, prédiction et probabilités de chaque sonde.
- **Ablation du contexte du BiGRU** (`scripts/ablation_sequence.py`) :
  contexte à la prédiction (nuit, 1 h, 5 min, 1 époque) et ordre mélangé.
- **Analyse des erreurs** (`scripts/analyse_errors.py`) à partir des
  prédictions sauvegardées : confusions, kappa par nuit, précision selon la
  distance au changement de stade, détail de N1.
- **AUPRC** (macro et par stade) et ROC-AUC par stade, dans la sortie et le
  JSON. La sortie compare la baseline officielle au tableau 8 de l'article
  SleepFM 2024 (PhysioNet 2018), stade par stade.

### Corrigé
- **Export des sondes** (`--save-models`) : le MLP, qui a aussi un attribut
  `model_`, était pris pour le BiGRU et l'export plantait. Seul un module
  torch est désormais déplacé sur CPU ; le test d'export couvre le MLP.
- **XGBoost sur Mac** : il plantait (segmentation fault) ou se bloquait
  dès que torch avait calculé dans le même processus (deux OpenMP
  différents, même sur un seul thread). `run_baseline.py` le retire des
  sondes sur Mac avec un avertissement ; ses tests sont sautés sur Mac et
  tournent en CI Linux.
- **Chargement OSF strict** : poids lus avec `weights_only=True`, et une clé
  manquante ou inattendue arrête le chargement (`allow_key_mismatch=True`
  pour un essai explicite).
- **Découpage figé validé** : sujet en double ou à la fois en train et en
  test refusé ; écriture atomique ; les nouveaux sujets ramènent la part du
  test à la fraction voulue (ajoutés un par un, ils allaient tous au train).
- **Caches vérifiés** : dimension, valeurs finies, stades connus, numéros
  d'époque uniques et temps croissants ; un cache invalide arrête le run.
- **Fin d'extraction** : fichiers de stades et de temps écrits de façon
  atomique et relus avant de sauter un sujet ; `--retry-skipped` retente les
  sujets rejetés.
- **Trous d'enregistrement** : le BiGRU et l'analyse d'erreurs coupent la
  nuit quand deux époques consécutives sont séparées de plus de 31 s (temps
  d'origine `onset_s`) ; sans ces temps (caches actuels), la nuit est
  supposée continue, avec un avertissement.
- **Scripts d'analyse** (ablation, BiGRU pondéré) : même chargement vérifié
  que le run principal (manifeste, découpage figé, caches).
- **Notebook** : s'arrête en erreur si le run n'a rien produit (le `| tee`
  masquait le code d'erreur).
- **Doc** : un seul run de référence dans le README ; formulations
  nuancées (checkpoint public non certifié identique à celui de l'article,
  « MLP essayé » plutôt que « frontière non linéaire »).
- **Début utile de l'enregistrement** (prétraitement version 2) : fin de la
  période « undefined » initiale, colonne désignée par son nom, comme le code
  officiel, au lieu du premier changement de la dernière colonne (« wake »).
  Identique sur les enregistrements vérifiés ; les caches actuels restent en
  version 1.
- **Extraction interrompue** : un dossier d'époques partiel n'est plus pris
  pour un sujet terminé ; le sujet est réextrait.
- **BiGRU** : le rembourrage des fenêtres courtes passait dans le GRU et
  influençait les vraies époques (sens retour). Le GRU reçoit maintenant la
  longueur réelle de chaque fenêtre. Effet mesuré négligeable (κ 0,6845 →
  0,6853).
- **Référence à l'article** : la documentation affirmait que SleepFM ne
  publiait pas de résultat sur PhysioNet 2018 et que la comparaison
  attendait MESA. L'article SleepFM 2024 évalue pourtant PhysioNet 2018
  (section 4.6, tableau 8 : ROC-AUC 0,924, AUPRC 0,759, F1 0,700). MESA
  devient une évaluation supplémentaire, pas un préalable.
- La formule « le test n'a servi qu'une fois » est précisée : le test ne sert
  à aucun choix, mais les mêmes nuits ont été évaluées par plusieurs runs.
- **Baseline** : le protocole officiel (`sleepfm/utils.py::train_model`)
  entraîne `LogisticRegression(class_weight="balanced")`. La baseline
  `logistic` est désormais pondérée ; l'ancienne, non pondérée, devient
  `logistic_unweighted`. Les résultats de la 0.1.0 venaient de la version
  non pondérée.
- **Convergence** : `max_iter` passe à 1 000 et la convergence de la
  régression logistique est vérifiée et affichée ; le notebook ne masque
  plus les avertissements.
- **Arrêt précoce du MLP** sur des nuits mises à part, au lieu d'époques
  tirées au hasard dans les mêmes nuits que l'entraînement (fuite).
- Chargement du checkpoint avec `weights_only=True`.
- **Reprise du mode par lots** : un `.npz` interrompu en pleine écriture
  restait sur le disque (ou le Drive), était pris pour un sujet traité et
  faisait planter le run suivant. L'écriture passe par un fichier
  temporaire renommé ensuite ; un cache illisible existant est détecté,
  supprimé et recalculé au démarrage.

### Résultats
- **500 sujets, 400 / 100 nuits** (2026-11-05), même nombre de nuits de test
  que l'article : baseline pondérée κ 0,599 [0,570 ; 0,627], non pondérée
  0,613, MLP 0,591, BiGRU 0,664 [0,634 ; 0,694] (+0,065 [+0,047 ; +0,083],
  81 nuits sur 100). Article : ROC-AUC 0,921 / AUPRC 0,750 / F1 0,686 contre
  0,924 / 0,759 / 0,700. Sur les 40 nuits de test du run à 200 sujets,
  l'entraînement sur 400 nuits fait mieux (baseline 0,616 contre 0,587,
  BiGRU 0,672 contre 0,656).
- 200 sujets, découpage figé étendu à 160 / 40 (2026-11-05), embeddings en
  prétraitement v2 calculés sur Kaggle : baseline pondérée κ 0,587
  [0,533 ; 0,643], non pondérée 0,608, MLP 0,586, BiGRU 0,656
  [0,594 ; 0,713] (+0,069 [+0,045 ; +0,096] sur la baseline, 33 nuits sur
  40). Sur les 20 anciennes nuits de test, l'entraînement sur 160 nuits fait
  mieux qu'à 100 sujets (baseline 0,620 contre 0,607, BiGRU 0,706 contre
  0,685) ; la baisse globale vient de deux nouvelles nuits à l'hypnogramme
  atypique (κ 0,04 et 0,12), gardées dans le test. Article : ROC-AUC 0,916 /
  AUPRC 0,750 / F1 0,679 contre 0,924 / 0,759 / 0,700.
- Les deux nuits atypiques ont des signaux propres : tr03-1292 a un EEG de
  sommeil sur 557 époques notées en éveil (annotation à vérifier),
  tr04-0133 un sommeil léger et fragmenté noté en N1.
- Embeddings v2 contre v1 sur les 100 premières nuits : identiques sur 91 ;
  sur les 9 autres, la v2 récupère 480 époques de début de nuit, mêmes
  stades sur les époques communes.
- Comparaison à l'article SleepFM 2024 (tableau 8, PhysioNet 2018) :
  baseline officielle ROC-AUC 0,925 / AUPRC 0,770 / F1 0,683, contre
  0,924 / 0,759 / 0,700 publiés. Le profil du F1 par stade de la logistique
  non pondérée est le plus proche de l'article.
- Ablation du BiGRU sur 3 graines : gain de κ sur la baseline de +0,060 à
  +0,077 avec la nuit entière (toujours significatif), +0,03 à +0,04 avec
  l'ordre mélangé. L'ordre chronologique ajoute +0,02 à +0,04 en comparaison
  directe, significatif sur une graine sur trois : sa part exacte n'est pas
  établie (l'estimation « la moitié du gain » d'une seule graine est retirée).
  F1 de N1 du BiGRU : 0,46 à 0,48 selon la graine, pas de gain établi.
- Erreurs du BiGRU : gain surtout sur N2 et au milieu des segments stables,
  presque nul aux changements de stade ; rappel de N1 en baisse (0,47 contre
  0,53), plus souvent lissé en éveil.
- BiGRU pondéré : rappel de N1 0,58 (contre 0,47), F1 de N1 inchangé, κ
  0,629, soit −0,056 [−0,083 ; −0,033] face au BiGRU non pondéré, qui reste
  la référence.
- 100 sujets, split 80/20 par sujet, protocole officiel (2026-11-04) :
  baseline pondérée κ 0,606 [0,545 ; 0,669], non pondérée 0,646, MLP 0,623,
  BiGRU 0,684 [0,615 ; 0,739]. XGBoost éliminé par la sélection (gap F1
  +0,32). Mêmes chiffres sur GPU (Kaggle) et CPU (Colab), à 0,001 près.

### Ajouté
- Comparaison appariée de chaque sonde à la baseline, sur les mêmes nuits de
  test (bootstrap sur les sujets) : écart de κ et de F1 macro avec son IC,
  dans la sortie et le JSON (`vs_baseline`). Gain du BiGRU significatif
  (κ +0,078 [+0,041 ; +0,117]), pas celui du MLP.
- Intervalles de confiance à 95 % du kappa et du F1 macro par bootstrap sur
  les sujets de test.
- Matrice de confusion de la baseline (sortie et JSON).
- Notebook utilisable aussi sur Kaggle (détection automatique, import des
  embeddings depuis un Dataset Kaggle, résultats dans le panneau Output) ;
  il force aussi la mise à jour du code à la relance.
- Le JSON de résultats enregistre les sujets de train et de test, les
  réglages (sondes, `max_iter`, plis, seuils de sélection) et la convergence
  de chaque sonde : deux runs se comparent sans relire les logs.

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
