# Des fenêtres à l'IAH ≥ 15 par nuit — 12 novembre 2026

**Les deux modèles classent correctement environ 3 nuits sur 4 selon l'IAH ≥ 15, mais aucun
ne domine.** Avec la calibration fixée sur la validation, SleepFM repère plus de nuits à IAH
élevé (sensibilité **83,3 %**, spécificité 73,9 %) ; OSF fait moins de fausses alertes
(sensibilité 68,5 %, spécificité **82,6 %**). L'indice d'OSF ordonne un peu mieux les nuits
(AUC 0,843 contre 0,807), sans écart démontré : **+0,036 [IC 95 % : −0,050 ; +0,131]**.
Résultat exploratoire : les 100 nuits de test avaient déjà servi à six analyses.

## Protocole

[Plan fixé avant calcul](../plans/Plan%20%C3%A9v%C3%A9nements%20et%20IAH%20SleepFM%20OSF%2012-11.md),
validé par le tuteur, avec ses précisions d'exécution. Un seul passage sur le test.

- **Référence** : IAH de l'étude de faisabilité (événements des 4 types par heure de sommeil
  expert), identique à `iah_500_sujets_ouverts.csv` pour les 140 nuits. Aucune nuit sous 1 h
  de sommeil : **40 nuits de validation et 100 de test utilisées**, dont **54 à IAH ≥ 15**
  en test.
- **Indice** : fenêtres de 30 s prédites positives par heure de fenêtres en cache (toutes,
  labels exclus compris), aux seuils initiaux (SleepFM 0,6934 ; OSF 0,7636). Modèles figés.
- **Calibration** sur les 40 nuits de validation, figée avant le test :
  SleepFM `IAH = 1,99 + 0,869·x` ; OSF `IAH = −1,90 + 1,025·x`.
- **Incertitude** : 1 000 tirages des nuits de test (graine 42), aucun tirage écarté.

## Classement IAH ≥ 15 (critère principal)

| 100 nuits de test | SleepFM | OSF |
|---|---|---|
| Sensibilité | **83,3 %** [72,9 ; 92,3] | 68,5 % [55,2 ; 80,7] |
| Spécificité | 73,9 % [60,0 ; 86,0] | **82,6 %** [71,4 ; 92,6] |
| Nuits IAH ≥ 15 repérées / manquées | 45 / 9 | 37 / 17 |
| Nuits IAH < 15 signalées à tort | 12 sur 46 | 8 sur 46 |
| AUC de l'indice (sans calibration) | 0,807 [0,705 ; 0,897] | 0,843 [0,757 ; 0,919] |

Le compromis sensibilité/spécificité dépend aussi de la calibration : la droite d'OSF place
moins de nuits au-dessus de 15. Sur 40 nuits, cette droite reste incertaine et
les IC ci-dessus ne couvrent pas cette incertitude.

## Secondaires

| 100 nuits de test | SleepFM | OSF |
|---|---|---|
| Erreur absolue moyenne sur l'IAH | 9,0 /h | 8,6 /h |
| Corrélation de Spearman | 0,70 | 0,72 |
| Biais de Bland-Altman [limites] | +0,8 [−24,3 ; +25,9] /h | −2,2 [−26,2 ; +21,7] /h |
| κ pondéré linéaire, 4 classes | 0,43 | 0,50 |
| Indice secondaire (somme des probabilités) : sensibilité / spécificité | 85,2 % / 63,0 % | 74,1 % / 76,1 % |

L'IAH estimé est trop imprécis pour situer une nuit dans sa classe : les limites de
Bland-Altman dépassent ±20 événements par heure.

## Événements (descriptif)

| 100 nuits de test | SleepFM | OSF |
|---|---|---|
| Apnées obstructives retrouvées | 64,7 % | 74,5 % |
| Apnées centrales | 54,0 % | 68,6 % |
| Apnées mixtes | 89,2 % | 96,6 % |
| Hypopnées | **57,5 %** | **51,3 %** |
| Tous événements | 59,8 % | 61,7 % |
| Séquences prédites sans aucun événement | 3 294 (4,4 par heure) | 2 310 (3,1 par heure) |

Un événement est retrouvé si les fenêtres positives en couvrent au moins 10 s ; une séquence
peut couvrir plusieurs événements. Les hypopnées restent le point faible, surtout pour OSF.

## Vérifications

- Probabilités recalculées sur 239 334 fenêtres déjà évaluées (validation et test, deux
  modèles) : écart maximal **1,7 × 10⁻⁶**, aucune fenêtre classée différemment au seuil. Un
  premier lancement s'était arrêté sur la validation avec la tolérance initiale de 10⁻⁶
  (archives en float32) ; tolérance portée à 10⁻⁵, plan modifié **avant toute lecture du
  test**, sans résultat écrit.
- Empreintes des modèles, caches, labels, annotations et plan vérifiées avant et après le
  calcul. Code commité avant le run (`52f5ee5`, `code_modified: false`).
- 9 tests unitaires (`tests/test_ahi.py`) : règle de l'IAH, calibration, AUC, κ, Bland-Altman.

## Limites

Même cohorte que le développement et nuits de test très explorées : aucune règle n'est
validée. PSG de laboratoire, pas polygraphie à domicile ; 54 % de nuits à IAH ≥ 15, plus
qu'en dépistage. Indice par heure d'enregistrement, pas de sommeil. Fenêtres de 30 s pour
des événements de 10 à 60 s. La confirmation se fera sur NSRR (décision du tuteur), avec la
règle figée ici ; les 494 sujets de confirmation ne sont pas utilisés.

## Fichiers

- [Résultats complets](../../results/respiratory_ahi_physionet2018.json)
- [Valeurs par nuit](../../results/respiratory_ahi_physionet2018.nights.npz)
- [Code et plan exécutés](../../results/respiratory_ahi_physionet2018.code.zip)
- Script : `scripts/estimate_respiratory_ahi.py` ; module : `src/data/ahi.py`
