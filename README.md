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

500 sujets PhysioNet 2018, découpage **par sujet** figé
([`configs/split_physionet2018.json`](configs/split_physionet2018.json) :
400 nuits d'entraînement, **100 de test** comme l'article, 89 132 époques de
test), embeddings SleepFM figés (prétraitement version 2). Run de référence
du 2026-11-05 ([`results/…_500`](results/results_physionet2018_500.txt), avec
les prédictions du test époque par époque ; embeddings calculés sur Kaggle,
sondes sur CPU). Les sujets des runs à 100 et 200 sujets gardent leur rôle.
Le test ne sert à aucun choix de modèle ou de réglage. IC à 95 % par
bootstrap sur les sujets de test.

| Sonde (test) | Accuracy | Cohen's κ [IC 95 %] | F1 macro [IC 95 %] | ROC-AUC | AUPRC | Gap F1 train − test |
|---|---|---|---|---|---|---|
| **Régression logistique pondérée** (baseline officielle) | 0,691 | **0,599** [0,570 ; 0,627] | 0,686 [0,664 ; 0,707] | 0,921 | 0,750 | +0,037 |
| Régression logistique non pondérée | 0,716 | 0,613 [0,581 ; 0,641] | 0,688 [0,666 ; 0,708] | 0,921 | 0,751 | +0,044 |
| MLP | 0,704 | 0,591 [0,558 ; 0,621] | 0,666 [0,643 ; 0,688] | 0,917 | 0,736 | +0,036 |
| **BiGRU sur la nuit** | **0,753** | **0,664** [0,634 ; 0,694] | **0,732** [0,708 ; 0,752] | **0,940** | **0,800** | +0,040 |

AUPRC = *average precision* (scikit-learn), un stade contre tous, moyennée.
Mêmes sondes qu'aux runs à 100 et 200 sujets (XGBoost, éliminé à 100
sujets, n'est pas relancé ; il ne tourne pas sur Mac, voir le
[guide](docs/GUIDE.md)).

- **Le contexte de la nuit est le levier** : le gain du BiGRU sur la baseline
  est **significatif** en comparaison appariée (mêmes nuits, bootstrap sur les
  sujets) : κ +0,065 [+0,047 ; +0,083], F1 macro +0,046 [+0,031 ; +0,062].
  Il progresse sur 81 nuits de test sur 100. Le MLP n'apporte rien : κ
  −0,008 [−0,027 ; +0,012], F1 macro −0,019.
- **Avec 100 nuits de test, les intervalles se resserrent** : ±0,03 sur le κ,
  contre ±0,06 à 40 nuits.
- **Plus de données d'entraînement aide**
  ([détail](results/nuits_test_physionet2018_500.txt)). Sur les 40 nuits de
  test du run à 200 sujets, entraîner sur 400 nuits au lieu de 160 fait
  mieux : κ 0,616 contre 0,587 pour la baseline, 0,672 contre 0,656 pour le
  BiGRU. Le gap train − test baisse aussi (+0,04 contre +0,06). Même constat
  entre 100 et 200 sujets ([détail](results/nuits_test_physionet2018_200.txt)).
- **Quelques nuits difficiles pèsent sur le total** : 5 nuits sur 100 ont un
  κ sous 0,3 avec la baseline (κ médian par nuit 0,583). Les deux plus
  basses, tr03-1292 (69 % d'éveil selon l'expert, lu surtout en N2) et
  tr04-0133 (76 % de N1), ont des signaux propres
  ([examen](results/nuits_atypiques_physionet2018_200.txt)). Dans tr03-1292,
  les époques notées en éveil et lues en N2 ont un EEG de sommeil (80 % de
  la puissance en delta et thêta, presque pas d'alpha ni de bêta, tonus du
  menton bas) : annotation à vérifier, ou EEG lent pathologique. Dans
  tr04-0133, le N1 est un sommeil léger et fragmenté, cas limite de
  cotation. À montrer au médecin du sommeil. Toutes les nuits restent dans
  le test : on n'écarte pas de nuits après avoir vu les résultats.

Les analyses détaillées ci-dessous portent sur le run à 100 sujets du
2026-11-04 ([`results/…_100_auprc`](results/results_physionet2018_100_auprc.txt) :
80 / 20 nuits, baseline κ 0,607, BiGRU κ 0,685, écart apparié +0,078
[+0,042 ; +0,114]) ; elles n'ont pas été refaites à 500 sujets.
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
  et sur CPU (Colab, Mac), à 0,001 près. Les embeddings recalculés en
  prétraitement v2 sont identiques à ceux de la v1 sur 91 des 100 premières
  nuits ; sur les 9 autres, la v2 récupère 480 époques de début de nuit que
  la v1 coupait, avec les mêmes stades sur les époques communes.

Sorties brutes dans [`results/`](results/) : texte, JSON (sujets, réglages,
matrices de confusion), prédictions du test, ablation (avec ses prédictions)
et analyse d'erreurs. Runs antérieurs gardés pour l'historique :
`…_200` (160 / 40), `…_100_auprc` (référence à 100 sujets), `…_100`
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
| Logistique pondérée (baseline officielle) | 0,921 | 0,750 | 0,686 | 0,77 / 0,50 / 0,72 / 0,74 / 0,70 |
| Logistique non pondérée | 0,921 | 0,751 | 0,688 | 0,77 / 0,45 / 0,78 / 0,74 / 0,70 |
| BiGRU sur la nuit | 0,940 | 0,800 | 0,732 | 0,80 / 0,50 / 0,81 / 0,75 / 0,80 |

- **Au niveau de l'article, avec le même nombre de nuits de test (100)** :
  ROC-AUC 0,921 contre 0,924, AUPRC 0,750 contre 0,759, F1 macro 0,686
  contre 0,700 pour la baseline officielle. Par stade, ROC-AUC et AUPRC sont
  au niveau publié à ±0,02 ; l'écart de F1 vient surtout de N2 (0,72 contre
  0,79), compensé en partie par N1 (0,50 contre 0,44). Le BiGRU dépasse
  l'article sur les trois métriques macro.
- **Le profil du F1 de la logistique non pondérée colle à celui de
  l'article**, stade par stade (N1 0,45 contre 0,44 ; N2 0,78 contre 0,79).
  La pondérée s'en écarte : meilleure sur N1, moins bonne sur N2. C'est une
  hypothèse à vérifier : l'évaluation de l'article n'a peut-être pas pondéré
  les classes, alors que le code officiel le fait.
- **Pas une reproduction stricte** : même nombre de nuits de test (100), mais
  notre propre découpage, et l'article ne précise ni le nombre de nuits d'entraînement ni
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
