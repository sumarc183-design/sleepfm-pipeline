# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [Non publié]

### Corrigé
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
- 100 sujets, split 80/20 par sujet, protocole officiel (2026-11-04) :
  baseline pondérée κ 0,606 [0,545 ; 0,669], non pondérée 0,646, MLP 0,623,
  BiGRU 0,684 [0,615 ; 0,739]. XGBoost éliminé par la sélection (gap F1
  +0,32). Mêmes chiffres sur GPU (Kaggle) et CPU (Colab), à 0,001 près.

### Ajouté
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
