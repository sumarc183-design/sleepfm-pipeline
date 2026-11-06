# Protocole de confirmation SleepFM

**Version 2, 2026-11-06 (version 1 : 2026-11-05), avant d'ouvrir les sujets réservés. État : à valider avec le tuteur.**
Toute modification se fait par un commit daté, avant l'évaluation. L'historique git fait foi.

## Pourquoi

Le run à 500 sujets est une **évaluation élargie avec réutilisation partielle du test** : sur ses
100 nuits de test, 40 avaient déjà été examinées, dont 20 sur plusieurs runs et ablations, et aucun
protocole n'avait été écrit à l'avance. Ce document fixe une confirmation sur des sujets jamais
ouverts. Il ne fait pas de ce travail une reproduction stricte de l'article : sujets, poids exacts
et sonde de l'article restent inconnus ou non alignés.

## Question

Sur des sujets jamais ouverts, le BiGRU exporté fait-il mieux que la baseline officielle exportée ?

## Données

- **Sujets réservés** : les 494 sujets d'entraînement PhysioNet 2018 absents de
  `configs/split_physionet2018.json`, de `tr06-0815` à `tr14-0291`, figés dans
  `configs/reserved_physionet2018.json` (identifiants seuls, d'après le fichier RECORDS de PhysioNet).
  Aucun embedding n'en a été calculé.
- **Limite** : ce ne sont pas des sujets tirés au hasard parmi les 994. Les 500 sujets déjà utilisés
  sont les premiers par identifiant (`tr03` à `tr06-0812`), et la signification des préfixes n'a pas
  été établie. Une baisse sur ces sujets pourrait donc venir d'un changement de lot.
- **Stades** : annotations d'origine, jamais modifiées.
- **Exclusions** : seulement les rejets automatiques de l'extraction, avec leur raison enregistrée.
  Aucun retrait après lecture d'un score.

## Modèles figés

Aucun entraînement : les exports locaux `models/physionet2018_500/` (graine 42, 400 nuits, contrat
checkpoint `a2fe7e5a…`, prétraitement version 2), identifiés par leur SHA-256. La graine 42 est gardée même
si sur 3 graines son κ est le plus bas (0,664 ; 0,673 ; 0,682) : le choix ne dépend d'aucun résultat sur
les sujets réservés. Ces exports sont au format 1 et ne sont pas réexportés, pour garder leurs empreintes :
ils n'ont ni les paramètres du MLP ni les nuits de validation du format 2, sans effet sur leurs prédictions.

| Fichier | SHA-256 |
|---|---|
| `logistic.joblib` | `1e219b2e5e485f82bd990319234b9316c369127b577ba592870092807ee3cfc0` |
| `sequence.joblib` | `85936d27eb2e136208113a53c61697f2005287ae4dcac084be601e8a2c861fe8` |

Liste, découpage, contrat et empreintes sont figés dans `configs/confirmation_physionet2018.json` :

| Fichier | SHA-256 |
|---|---|
| `configs/reserved_physionet2018.json` | `0903cff2fba7176507c5ae9094e302fe999b5e3b7cf644e2e328c185a0a1043b` |
| `configs/split_physionet2018.json` | `d4b33aa2aacd556b2f248883a41fe9d9d065e6a8a458d005e5d642dd9554ef72` |

## Exécution

1. Embeddings seuls des 494 sujets, dans un dossier à part, sans sonde (notebook avec `CONFIRMATION = True`,
   voir le [guide](GUIDE.md#confirmation-sur-les-sujets-réservés)). C'est le premier accès à ces signaux :
   seulement après validation de ce protocole.
2. `python scripts/confirm_frozen_models.py --embeddings-dir embeddings/physionet2018_reserved_v2 --out results/confirmation_physionet2018`.
   Elle refuse tout écart aux empreintes, au contrat ou à la couverture des sujets, n'entraîne rien et ne
   calcule aucun score en cas de refus. Elle a été répétée uniquement sur les 100 nuits de test déjà ouvertes.

## Analyse prévue

| Rôle | Mesure |
|---|---|
| **Critère principal** | Écart de κ, BiGRU moins baseline, bootstrap apparié sur les sujets, 1 000 tirages, graine 42 |
| **Conclusion** | Gain confirmé si la borne basse de l'IC à 95 % est au-dessus de 0. Sinon, gain non confirmé |
| Secondaire | Écart de F1 macro ; κ, F1 macro, ROC-AUC et AUPRC (*average precision*) de chaque modèle ; F1 par stade |
| Secondaire | Comparaison au run à 500 sujets et κ par préfixe d'identifiant, à titre descriptif |

Un seul passage. Le résultat est publié quel qu'il soit, avec la commande exacte, le commit et ce protocole.

## Règles pour la suite

- Une nouvelle variante se choisit sur une validation interne, à partir des sujets d'entraînement.
  Les sujets réservés ne servent ni à choisir ni à régler.
- Une variante retenue après cette confirmation demande de nouveaux sujets non ouverts, ou se présente
  comme exploratoire.
- Les analyses de variance entre graines et d'effet du volume se font sur les sujets déjà ouverts :
  elles restent exploratoires.

## Historique d'utilisation du test

| Nuits | Utilisation |
|---|---|
| 20 nuits du run à 100 sujets | Plusieurs runs, ablation sur 3 graines, analyse d'erreurs, BiGRU pondéré, examen de nuits |
| 20 nuits ajoutées à 200 sujets | Run à 200 sujets, comparaison par nuit, examen de 2 nuits |
| 60 nuits ajoutées à 500 sujets | Run à 500 sujets, comparaison par nuit |
| Les 100 nuits (2026-11-06) | Ablation par capteurs, graines 42 à 44, effet du volume (160 ou 400 nuits), variance des blocs, répétition de la commande de confirmation |
| 494 sujets réservés | Jamais ouverts |
