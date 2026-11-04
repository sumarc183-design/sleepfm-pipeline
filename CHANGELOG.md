# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté
- **Découpage figé** (`--split-file`, `configs/split_physionet2018.json`) :
  un sujet garde son rôle quand on en ajoute ; les 80 / 20 sujets actuels y
  sont enregistrés.
- **Manifeste du cache d'embeddings** (`manifest.json`) : empreinte du
  checkpoint, version du prétraitement, canaux, fréquence. Un cache d'un
  autre checkpoint est refusé, un autre prétraitement n'est jamais mélangé.
  Enregistré dans le JSON des résultats.
- **Temps d'origine de chaque époque** (`onset_s`) dans l'extraction et le
  cache ; **raison des rejets** enregistrée.
- **Notebook** : 200 sujets par défaut, dans `physionet2018_v2` avec le
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
- Comparaison à l'article SleepFM 2024 (tableau 8, PhysioNet 2018) :
  baseline officielle ROC-AUC 0,925 / AUPRC 0,770 / F1 0,683, contre
  0,924 / 0,759 / 0,700 publiés. Le profil du F1 par stade de la logistique
  non pondérée est le plus proche de l'article.
- Ablation du BiGRU : κ +0,078 sur la baseline avec la nuit entière, +0,070
  avec 1 h, +0,039 avec 5 min ; +0,041 avec l'ordre mélangé. Environ la
  moitié du gain vient de l'ordre chronologique.
- Erreurs du BiGRU : gain surtout sur N2 et au milieu des segments stables,
  presque nul aux changements de stade ; rappel de N1 en baisse (0,47 contre
  0,53), plus souvent lissé en éveil.
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
