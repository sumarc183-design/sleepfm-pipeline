# SleepFM Pipeline

[![CI](https://github.com/sumarc183-design/sleepfm-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/sumarc183-design/sleepfm-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)
[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumarc183-design/sleepfm-pipeline/blob/main/notebooks/sleepfm_embeddings_physionet2018.ipynb)
[![Ouvrir dans Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/sumarc183-design/sleepfm-pipeline/blob/main/notebooks/sleepfm_embeddings_physionet2018.ipynb)

Reproduction et évaluation de [SleepFM](https://github.com/rthapa84/sleepfm-codebase),
modèle fondamental multimodal du sommeil, sur données ouvertes : stadification
en 5 classes (Wake, N1, N2, N3, REM) à partir des embeddings figés du
checkpoint officiel, avant toute comparaison à
[OSF](https://github.com/yang-ai-lab/OSF-Open-Sleep-FM) et tout développement
propriétaire.

```mermaid
flowchart LR
    A[PSG PhysioNet 2018<br/>13 canaux] --> B[Époques de 30 s<br/>256 Hz]
    B --> C[SleepFM figé<br/>3 encodeurs EffNet]
    C --> D[Embeddings<br/>respiratoire + stades + ECG]
    D --> E[Sondes<br/>logistique officielle · non pondérée · MLP/XGBoost · BiGRU]
    E --> F[Accuracy · κ · F1 · ROC-AUC<br/>gap train − test]
```

## Résultats

100 sujets PhysioNet 2018, découpage **par sujet** figé
([`configs/split_physionet2018.json`](configs/split_physionet2018.json) :
80 nuits d'entraînement, 20 de test, 17 818 époques de test), embeddings
SleepFM figés. Run de référence du 2026-11-04
([`results/…_auprc`](results/results_physionet2018_100_auprc.txt), avec les
prédictions du test époque par époque). Le test ne sert à aucun choix de
modèle ou de réglage, mais les mêmes 20 nuits ont été évaluées par plusieurs
runs : les analyses ci-dessous restent exploratoires. IC à 95 % par bootstrap
sur les sujets de test.

| Sonde (test) | Accuracy | Cohen's κ [IC 95 %] | F1 macro [IC 95 %] | ROC-AUC | AUPRC | Gap F1 train − test |
|---|---|---|---|---|---|---|
| **Régression logistique pondérée** (baseline officielle) | 0,694 | **0,607** [0,545 ; 0,669] | 0,683 [0,629 ; 0,729] | 0,925 | 0,770 | +0,069 |
| Régression logistique non pondérée | 0,737 | 0,645 [0,577 ; 0,702] | 0,697 [0,642 ; 0,741] | 0,927 | 0,771 | +0,064 |
| MLP (retenu face à XGBoost) | 0,714 | 0,623 [0,556 ; 0,684] | 0,683 [0,629 ; 0,726] | 0,924 | 0,762 | +0,084 |
| **BiGRU sur la nuit** | **0,765** | **0,685** [0,613 ; 0,740] | **0,739** [0,674 ; 0,784] | **0,936** | **0,801** | +0,055 |

AUPRC = *average precision* (scikit-learn), un stade contre tous, moyennée.

- **Le contexte de la nuit est le levier** : le gain du BiGRU sur la baseline
  est **significatif** en comparaison appariée (mêmes nuits, bootstrap sur les
  sujets) : κ +0,078 [+0,042 ; +0,114], F1 macro +0,056 [+0,026 ; +0,085].
  Celui du MLP ne l'est pas : κ +0,016 [−0,006 ; +0,038].
- **Il tient d'une graine à l'autre, avec une marge**
  ([ablation, 3 graines](results/ablation_sequence_physionet2018_100.txt)) :
  κ du BiGRU de 0,666 à 0,684, gain de +0,060 à +0,077, significatif à
  chaque fois. Le 0,685 du run de référence est donc plutôt haut ; compter
  environ ±0,01 de variation due à l'entraînement.
- **D'où vient le gain** : il grandit avec le contexte lu à la prédiction
  (+0,02 à +0,03 avec 5 min, +0,06 à +0,07 avec 1 h, +0,06 à +0,08 avec la
  nuit). Avec les époques mélangées dans chaque nuit, il reste +0,03 à +0,04 :
  voir toute la nuit du sujet aide déjà. L'ordre chronologique ajoute
  +0,02 à +0,04 en comparaison directe, significatif sur une graine sur trois
  seulement : sa part exacte n'est pas établie. Le BiGRU lit aussi les
  époques futures : analyse après l'enregistrement, pas temps réel.
- **Où le BiGRU gagne** ([analyse des erreurs](results/errors_sequence_vs_logistic_physionet2018_100.txt)) :
  surtout sur N2, que la baseline confond avec N3, N1 et REM (N2 bien classé :
  5 826 époques contre 4 300). Le gain se fait au milieu des segments stables
  (+0,07 à +0,10 de précision), presque pas aux changements de stade (+0,01),
  qui restent le point dur (55 % de précision, 21 % des époques). 17 nuits de
  test sur 20 progressent (gain médian de κ +0,06).
- **N1 : pas de gain établi.** Dans le run de référence, F1 0,52 contre 0,51
  pour la baseline, mais 0,46 à 0,48 sur les 3 graines de l'ablation. Le
  rappel baisse (0,47 contre 0,53) : le BiGRU lisse davantage les vraies
  époques N1 en éveil (873 contre 718).
- **Deux nuits ratées par les deux modèles** (κ ≈ 0,34,
  [examen](results/nuits_ratees_physionet2018_100.txt)) : presque tout leur
  REM est pris pour de l'éveil. Dans l'une, l'EMG du menton est
  quasi plat (électrode probable) ; dans l'autre, le REM ressemble à l'éveil
  (peu d'atonie, pas de mouvements oculaires), à montrer au médecin du
  sommeil.
- **Pondérer les stades dans le BiGRU ne paie pas**
  ([essai](results/bigru_pondere_physionet2018_100.txt)) : le rappel de N1
  remonte (0,58 contre 0,47) mais son F1 ne bouge pas (0,52), N2 recule
  (0,74 contre 0,83) et le κ perd 0,056 [−0,083 ; −0,033]. Même compromis que
  pour la régression logistique : le BiGRU reste non pondéré.
- **La ROC-AUC varie peu** (0,924 à 0,936) : les probabilités classent bien
  les époques pour toutes les sondes ; elles se distinguent sur la décision
  finale, que mesurent le κ et le F1.
- **Le MLP essayé n'apporte rien** : il fait jeu égal avec la régression
  logistique (une autre architecture non linéaire n'est pas exclue). XGBoost,
  un peu moins bon en validation croisée (F1 0,669 contre 0,672), est en plus
  éliminé pour surapprentissage (gap F1 +0,32).
- **Pondérer les classes déplace les erreurs** : la baseline officielle
  reconnaît mieux N1 (F1 0,51 contre 0,45), mais confond davantage N2 avec
  N3, N1 et REM, d'où un κ plus bas (0,607 contre 0,645).
- **N1 reste le stade le plus difficile** (F1 ≈ 0,51), confondu surtout avec
  l'éveil (718 époques sur 2 571) et le REM (308). C'est aussi le stade le
  moins concordant entre experts humains.
- **Reproductible** : un même run donne les mêmes chiffres sur GPU (Kaggle)
  et sur CPU (Colab, Mac), à 0,001 près.

Sorties brutes dans [`results/`](results/) : texte, JSON (sujets, réglages,
matrices de confusion), prédictions du test, ablation (avec ses prédictions)
et analyse d'erreurs. Runs antérieurs gardés pour l'historique : `…_100`
(Kaggle), `…_cpu`, `…_paired`.

Avant le 2026-11-04 : κ 0,584 (20 sujets) puis 0,622 (100 sujets, split
75/25), avec la régression **non pondérée**, qui n'est pas le protocole
officiel ([détail](results/results_physionet2018_100_v0.1.0_non_pondere.txt)).

### Comparaison à l'article

L'article SleepFM 2024 ([Thapa et al., ICML 2024](https://arxiv.org/abs/2405.17766),
section 4.6, tableau 8) évalue SleepFM sur PhysioNet 2018, en validation
externe, avec 100 participants en test. Nous utilisons le checkpoint public
`best.pt` du dépôt officiel ; rien ne garantit qu'il soit identique aux
poids de l'article (son empreinte SHA-256 est enregistrée dans le manifeste
du cache). Ses résultats principaux
portent sur les données internes de Stanford, non publiques ; MESA n'y
apparaît pas. Il ne publie pas de κ sur PhysioNet 2018.

| Test | ROC-AUC macro | AUPRC macro | F1 macro | F1 Wake / N1 / N2 / N3 / REM |
|---|---|---|---|---|
| Article (tableau 8) | 0,924 | 0,759 | 0,700 | 0,79 / 0,44 / 0,79 / 0,74 / 0,72 |
| Logistique pondérée (baseline officielle) | 0,925 | 0,770 | 0,683 | 0,77 / 0,51 / 0,71 / 0,75 / 0,68 |
| Logistique non pondérée | 0,927 | 0,771 | 0,697 | 0,77 / 0,45 / 0,81 / 0,76 / 0,70 |
| BiGRU sur la nuit | 0,936 | 0,801 | 0,739 | 0,76 / 0,52 / 0,83 / 0,79 / 0,80 |

- **Même ordre de grandeur que l'article** : ROC-AUC et AUPRC macro au niveau
  publié, ou légèrement au-dessus.
- **Le profil du F1 de la logistique non pondérée colle à celui de
  l'article**, stade par stade (N1 0,45 contre 0,44 ; N2 0,81 contre 0,79).
  La pondérée s'en écarte : meilleure sur N1, moins bonne sur N2. C'est une
  hypothèse à vérifier : l'évaluation de l'article n'a peut-être pas pondéré
  les classes, alors que le code officiel le fait.
- **Pas une reproduction stricte** : 20 nuits de test contre 100, notre propre
  découpage, et l'article ne précise ni le nombre de nuits d'entraînement ni
  la sonde utilisée pour cette évaluation.

Chiffres du run de référence ci-dessus. L'AUPRC de l'article n'indique pas
sa méthode de calcul ; la nôtre est l'*average precision*. La sortie du script
compare la baseline au tableau 8 stade par stade. Sondes et règle de
sélection : [guide](docs/GUIDE.md#sondes-et-sélection-de-modèle).

## Démarrage rapide

**Sans installation** : bouton *Open in Colab* ci-dessus, `GPU T4`, `Tout exécuter`. Quota Colab épuisé : bouton *Open in Kaggle*, même notebook ([détails](docs/GUIDE.md#kaggle-autre-gpu-gratuit)).

**En local** :

```bash
pip install -r requirements-dev.txt
bash scripts/setup_sleepfm.sh   # checkpoint officiel best.pt
python scripts/download_physionet2018.py --out data/raw/physionet2018/training --limit 20
python run_baseline.py --raw-data data/raw/physionet2018/training \
    --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt
pytest tests/                   # unitaires + bout-en-bout sur données synthétiques
```

Mode par lots pour 100 sujets et plus, détails des sondes, tests et points
d'attention : [docs/GUIDE.md](docs/GUIDE.md).

## Données

| Base | Accès | Usage |
|---|---|---|
| PhysioNet Challenge 2018 | Public (994 sujets annotés) | Validation du pipeline (phase 1) |
| MESA (NSRR) | Sur demande (DUA) | Évaluation sur une cohorte supplémentaire (phase 2) |
| SHHS (NSRR) | Sur demande (DUA) | Évaluation à plus grande échelle (phase 2) |
| CHAT (NSRR) | Sur demande (DUA) | Pédiatrie, piste de différenciation (phase 2) |

Procédure NSRR : [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

## Structure

```
run_baseline.py   point d'entrée : split par sujet -> extraction -> embeddings -> sondes -> évaluation
src/data/         lecture et découpage des PSG en époques
src/models/       encodeurs SleepFM, sondes (logistique, MLP, XGBoost, BiGRU), backbone OSF
scripts/          téléchargement PhysioNet, setup SleepFM, générateur de données synthétiques
notebooks/        notebook Google Colab / Kaggle
tests/            tests unitaires, bout-en-bout synthétique, données réelles
results/          sorties des runs réels
configs/          conventions de référence (non lues par les scripts)
docs/             guide, carte du modèle, roadmap, accès NSRR, comparaison OSF
```

## Documentation

| Document | Contenu |
|---|---|
| [GUIDE.md](docs/GUIDE.md) | Installation, modes d'exécution, sondes, tests, pièges connus |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Usage prévu, données, métriques, limites |
| [ROADMAP.md](docs/ROADMAP.md) | Jalons J30 / J60 / J90 |
| [DATA_ACCESS.md](docs/DATA_ACCESS.md) | Demande d'accès NSRR (MESA, SHHS, CHAT) |
| [OSF_COMPARISON.md](docs/OSF_COMPARISON.md) | Ce qu'il faut aligner avant de comparer SleepFM et OSF |
| [CHANGELOG.md](CHANGELOG.md) | Historique des versions |

## Attributions

Code de modèle et d'extraction adapté de SleepFM et d'OSF (licences MIT) :
voir [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Pour citer ce travail :
[CITATION.cff](CITATION.cff).

⚠️ Projet de recherche, pas un dispositif médical.
