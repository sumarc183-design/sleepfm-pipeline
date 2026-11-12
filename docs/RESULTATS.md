# Résultats détaillés

Résumé dans le [README](../README.md).

Tableau des scores déjà enregistrés : [Markdown](../results/consolidated_physionet2018.md)
et [CSV](../results/consolidated_physionet2018.csv), régénérés par
`scripts/consolidate_results.py` sans entraîner ni réévaluer de modèle. Les tests
et prétraitements diffèrent entre runs ; ce tableau ne constitue pas un classement.

## Run de référence à 500 sujets


**Évaluation élargie, avec réutilisation partielle du test : ce n'est pas une reproduction stricte
de l'article, ni une validation clinique.** 500 sujets PhysioNet 2018 (les premiers par identifiant,
pas un tirage au hasard parmi les 994), découpage **par sujet** figé
([`configs/split_physionet2018.json`](../configs/split_physionet2018.json) : 400 nuits d'entraînement,
100 de test, 89 132 époques de test), embeddings SleepFM figés (prétraitement version 2). Run de
référence du 2026-11-05 ([`results/…_500`](../results/results_physionet2018_500.txt), avec les prédictions
du test époque par époque ; embeddings calculés sur Kaggle, sondes sur CPU). IC à 95 % par bootstrap sur
les sujets de test : ils décrivent l'incertitude liée aux nuits pour ces modèles, pas la variabilité
entre entraînements ou centres.

Le test ne sert à aucun choix automatique de modèle, mais **40 des 100 nuits avaient déjà été examinées**
et aucun protocole n'avait été écrit avant ce run. La
[confirmation](PROTOCOLE_CONFIRMATION.md#résultat-2026-11-09-passage-unique) sur 494 sujets jamais ouverts,
faite le 2026-11-09 en un seul passage, confirme le gain : κ 0,672 contre 0,578, écart +0,094 [+0,086 ; +0,103].

| Sonde (test) | Accuracy | Cohen's κ [IC 95 %] | F1 macro [IC 95 %] | ROC-AUC | AUPRC | Gap F1 train − test |
|---|---|---|---|---|---|---|
| **Régression logistique pondérée** (baseline officielle) | 0,691 | 0,599 [0,570 ; 0,627] | 0,686 [0,664 ; 0,707] | 0,921 | 0,750 | +0,037 |
| Régression logistique non pondérée | 0,716 | 0,613 [0,581 ; 0,641] | 0,688 [0,666 ; 0,708] | 0,921 | 0,751 | +0,044 |
| MLP | 0,704 | 0,591 [0,558 ; 0,621] | 0,666 [0,643 ; 0,688] | 0,916 | 0,736 | +0,036 |
| **BiGRU sur la nuit** | **0,753** | **0,664** [0,634 ; 0,694] | **0,731** [0,708 ; 0,752] | **0,940** | **0,800** | +0,040 |

AUPRC = *average precision* (scikit-learn), un stade contre tous, moyennée. Mêmes sondes qu'aux runs à
100 et 200 sujets ; la sélection MLP / XGBoost n'a pas été refaite à 500 sujets (XGBoost, éliminé à 100
sujets, ne tourne pas sur Mac, voir le [guide](GUIDE.md)).

- **Le BiGRU fait mieux que la baseline sur ces 100 nuits** : en comparaison appariée (mêmes nuits,
  bootstrap sur les sujets), κ +0,065 [+0,047 ; +0,083], F1 macro +0,046 [+0,031 ; +0,062], et κ supérieur sur
  81 nuits sur 100. Le gain se retrouve à part sur les 60 nuits ajoutées (κ 0,659 contre 0,588) et sur les
  40 déjà vues (0,672 contre 0,616). Ce tableau est celui de la graine 42. Sur les graines 42, 43 et 44,
  le κ du BiGRU va de 0,664 à 0,682 (moyenne 0,673) et son gain de +0,065 à +0,083
  ([synthèse](../results/synthese_graines_volume_physionet2018_500.txt)) : 0,664 est la plus basse des trois.
- **Le MLP n'apporte pas de gain** : κ −0,008 [−0,027 ; +0,012], F1 macro −0,019 [−0,034 ; −0,003].
- **Effet du volume d'entraînement : modeste, mesuré sur le même test.** Même cache, mêmes 100 nuits de
  test, 160 nuits d'entraînement (les premières par ID, pas tirées au hasard) contre 400 : baseline κ
  +0,026 [+0,015 ; +0,037] ; BiGRU +0,012 [−0,005 ; +0,027], +0,019 [+0,002 ; +0,038] et
  +0,035 [+0,017 ; +0,053] selon la graine, +0,022 en moyenne
  ([synthèse](../results/synthese_graines_volume_physionet2018_500.txt)). Ces nuits ont déjà été examinées :
  c'est une mesure exploratoire.
- **Sur ces données, l'EEG/EOG porte presque toute la performance**
  ([script](../scripts/ablation_modalites.py), [synthèse](../results/synthese_graines_volume_physionet2018_500.txt)).
  κ moyen du BiGRU sur 3 graines : 0,673 avec tous les capteurs, 0,442 sans EEG/EOG (respiration + ECG).
  Avec la graine 42 seule, l'EEG/EOG seul donne 0,669. Le gain apparent sans ECG (0,687) n'est pas établi :
  +0,023, +0,013 et +0,006 selon la graine, IC contenant 0 pour deux d'entre elles. Exploratoire, encodeurs
  figés conçus pour la polysomnographie : ce n'est pas la limite de ces capteurs, et ça ne simule ni un
  téléphone ni un objet connecté. Le bloc respiration semble peu informatif, sans cause établie.
- **N1 reste difficile** : F1 0,500 pour la baseline, 0,505 pour le BiGRU. Pas de gain établi.
- **Quelques nuits pèsent sur le total** : 5 nuits sur 100 ont un κ sous 0,3 avec la baseline (κ médian
  par nuit 0,583). Sur les deux plus basses, tr03-1292 (κ 0,08, 69 % d'éveil selon l'expert, surtout lu en
  N2 par le BiGRU) et tr04-0133 (κ 0,24, 76 % de N1), l'expert et les modèles sont en désaccord
  ([mesures et hypothèses](../results/nuits_atypiques_physionet2018_500.txt)). Sur tr03-1292, les époques
  notées en éveil et lues en N2 ont 93 % de puissance lente en C3, contre 31 % pour celles où les deux
  s'accordent sur l'éveil. Ce sont des observations : ni une erreur d'annotation ni une erreur de modèle
  ne sont établies, et une relecture indépendante reste à faire. Toutes les nuits restent dans le test.

## Prédiction sur PSG brute, grille fixe

Mesure du 10 novembre 2026 (`results/raw_grid_physionet2018.json`, embeddings du commit
`e94dbd7`) : les 100 nuits de test, découpées en fenêtres de 30 s depuis le début de
l'enregistrement, **sans lire les annotations**, puis prédites par les modèles figés du
run à 500 sujets, sans entraînement.

| 100 nuits, 76 626 fenêtres notées | κ [IC 95 %] | F1 macro | Grille annotée, mêmes nuits (κ) |
|---|---|---|---|
| Baseline | 0,634 [0,603 ; 0,664] | 0,707 | 0,599 |
| BiGRU | **0,711** [0,678 ; 0,742] | **0,759** | 0,664 |

- **Fenêtres notées** : 76 626 sur 91 947 (83 %). Non notées : 12 606 à cheval sur deux
  stades et 2 715 hors annotation (début de l'enregistrement, en général). Les modèles
  prédisent toute la nuit ; seules ces fenêtres sont exclues du score.
- **Lecture** : sur les fenêtres d'un seul stade, la prédiction sur PSG brute ne perd rien.
  Le score est plus haut que sur la grille annotée, mais les deux ne se comparent pas
  directement : les fenêtres exclues sont surtout des transitions, les plus difficiles, que
  la grille annotée garde. **Le score de la grille brute est donc possiblement optimiste.**
- F1 par stade, baseline puis BiGRU : éveil 0,797 / 0,827 ; N1 0,496 / 0,520 ; N2 0,745 /
  0,837 ; N3 0,765 / 0,787 ; REM 0,730 / 0,825.
- **Limite** : les transitions ne sont pas mesurées sur cette grille ; nuits de test déjà
  ouvertes, même cohorte.
- **Complément du 12 novembre** : les fenêtres mixtes entièrement annotées sont réintégrées par
  majorité stricte (89 116 fenêtres notées) : BiGRU κ 0,664 [0,634 ; 0,694], baseline 0,600.
  Les 12 490 fenêtres réintégrées n'ont qu'un échantillon minoritaire (5 ms) en fin de fenêtre :
  ce ne sont pas des transitions cliniques. Détail : [fenêtres mixtes](resultats/grille_brute_transitions_2026-11-12.md)
  et [frontières de stade](resultats/frontieres_stade_2026-11-12.md).

## Exploration apnées/hypopnées (12 novembre 2026)

Sur les mêmes 100 nuits de test (partition 360/40/100, embeddings figés, sans les 494 sujets de
confirmation) : **OSF AP 0,661 contre 0,593 pour SleepFM**, écart **+0,068 [+0,023 ; +0,116]** ;
sensibilité 62,2 % contre 59,8 %. Les hypopnées concentrent 73,7 % des fenêtres positives manquées
par OSF. Exploratoire : cible regroupant plusieurs événements, même cohorte.
Détail : [baseline et comparaison OSF](resultats/respiratoire_baseline_osf_2026-11-12.md),
[analyse des erreurs](resultats/respiratoire_erreurs_2026-11-12.md) ; plans fixés avant calcul dans
[`plans/`](plans/).

**Compromis des seuils** (seuils choisis sur les 40 nuits de validation, appliqués au test) :

| Modèle | Seuil | Sensibilité test | Précision | Rappel hypopnées | Fausses fenêtres positives | Séquences de faux positifs / h évaluée |
|---|---|---:|---:|---:|---:|---|
| SleepFM | F1 initial | 59,8 % | 51,2 % | 54,1 % | 6 520 | 6,17 [5,18 ; 7,21] |
| SleepFM | Cible 80 % | 76,8 % | 38,1 % | 74,2 % | 14 300 | 11,13 [10,13 ; 12,17] |
| OSF | F1 initial | 62,2 % | 59,8 % | 48,9 % | 4 777 | 4,42 [3,62 ; 5,27] |
| OSF | Cible 80 % | 74,1 % | 49,0 % | 63,3 % | 8 817 | 7,23 [6,26 ; 8,18] |

Une cible de validation ne garantit pas la même sensibilité sur le test ; les deux modèles ne sont
donc pas comparés à rappel identique. Environ 70 % des séquences de faux positifs ne durent qu'une
fenêtre. Ce sont des fenêtres, pas des événements ni des alertes cliniques.
Détail : [seuils](resultats/respiratoire_seuils_2026-11-12.md) (cibles 70 et 90 % incluses),
[séquences](resultats/respiratoire_sequences_2026-11-12.md).

Diagnostics sur les seules 40 nuits de validation (aucune nouvelle lecture du test) :

- **Événements** : pour OSF au seuil initial, 72,5 % des événements évaluables sont recouverts,
  mais 33,5 % seulement sont appariés un à un aux séquences prédites
  ([compte rendu](resultats/respiratoire_evenements_validation_2026-11-12.md)).
- **Stabilité des seuils** : pour OSF à la cible 80 %, le rappel des nuits absentes du tirage
  va de 63,8 à 89,5 % ; le choix du seuil dépend nettement des nuits de calibration
  ([compte rendu](resultats/respiratoire_stabilite_seuils_2026-11-12.md)).

## Comparaison exploratoire avec OSF

Mêmes 100 nuits de test, mêmes fenêtres et mêmes sondes ; seule la représentation change
(critère fixé avant le calcul, `docs/OSF_COMPARISON.md`). Avec le BiGRU, **OSF obtient κ 0,717
contre 0,664 pour SleepFM, écart +0,052 [+0,032 ; +0,073]**, et fait mieux sur les 5 stades,
alors que 4 de ses 12 canaux manquent dans PhysioNet 2018. Résultat exploratoire, à confirmer
sur MESA ou SHHS : [détail](OSF_COMPARISON.md#option-a--résultat-passage-unique-critère-fixé-avant).

## Analyses détaillées du run à 100 sujets

Les analyses ci-dessous portent sur le run à 100 sujets du
2026-11-04 ([`results/…_100_auprc`](../results/results_physionet2018_100_auprc.txt) :
80 / 20 nuits, baseline κ 0,607, BiGRU κ 0,685, écart apparié +0,078
[+0,042 ; +0,114]) : prétraitement version 1, 20 nuits de test. Elles n'ont pas été refaites à 500 sujets.
- **Il tient d'une graine à l'autre, avec une marge**
  ([ablation, 3 graines](../results/ablation_sequence_physionet2018_100.txt)) :
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
- **Où le BiGRU gagne** ([analyse des erreurs](../results/errors_sequence_vs_logistic_physionet2018_100.txt)) :
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
  [examen](../results/nuits_ratees_physionet2018_100.txt)) : presque tout leur
  REM est pris pour de l'éveil. Dans l'une, l'EMG du menton est
  quasi plat (électrode probable) ; dans l'autre, le REM ressemble à l'éveil
  (peu d'atonie, pas de mouvements oculaires), à montrer au médecin du
  sommeil.
- **Pondérer les stades dans le BiGRU ne paie pas**
  ([essai](../results/bigru_pondere_physionet2018_100.txt)) : le rappel de N1
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

## Sorties et runs antérieurs

Sorties brutes dans [`results/`](../results/) : texte, JSON (sujets, réglages,
matrices de confusion), prédictions du test, ablation (avec ses prédictions)
et analyse d'erreurs. Runs antérieurs gardés pour l'historique :
`…_200` (160 / 40, prétraitement version 2), `…_100_auprc` (référence à 100 sujets), `…_100`
(Kaggle), `…_cpu`, `…_paired`.

Avant le 2026-11-04 : κ 0,584 (20 sujets) puis 0,622 (100 sujets, split
75/25), avec la régression **non pondérée**, qui n'est pas le protocole
officiel ([détail](../results/results_physionet2018_100_v0.1.0_non_pondere.txt)).

## Comparaison à l'article

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
| BiGRU sur la nuit | 0,940 | 0,800 | 0,731 | 0,80 / 0,50 / 0,81 / 0,75 / 0,80 |

- **Même ordre de grandeur que l'article, avec le même nombre de nuits de test (100)** : ROC-AUC 0,921
  contre 0,924, AUPRC 0,750 contre 0,759, F1 macro 0,686 contre 0,700 pour la baseline officielle. Par
  stade, l'écart de ROC-AUC et d'AUPRC reste sous 0,025 ; celui du F1 vient surtout de N2 (0,72 contre
  0,79), avec N1 au-dessus (0,50 contre 0,44).
- **Le BiGRU est numériquement au-dessus des valeurs publiées**, sur notre propre test : cela ne démontre
  pas une supériorité à protocole identique.
- **Le profil du F1 de la logistique non pondérée colle à celui de
  l'article**, stade par stade (N1 0,45 contre 0,44 ; N2 0,78 contre 0,79).
  La pondérée s'en écarte : meilleure sur N1, moins bonne sur N2. C'est une
  hypothèse à vérifier : l'évaluation de l'article n'a peut-être pas pondéré
  les classes, alors que le code officiel le fait.
- **Pas une reproduction stricte** : seul l'effectif de test est identique. Les sujets et le découpage
  sont les nôtres, 40 de nos nuits de test avaient déjà été examinées, et l'article ne précise ni le
  nombre de nuits d'entraînement ni la sonde utilisée pour cette évaluation.

Chiffres du run de référence ci-dessus. L'AUPRC de l'article n'indique pas
sa méthode de calcul ; la nôtre est l'*average precision*. La sortie du script
compare la baseline au tableau 8 stade par stade. Sondes et règle de
sélection : [guide](GUIDE.md#sondes-et-sélection-de-modèle).
