# Amplitude des sangles et du débit sur PhysioNet 2018 — 13 novembre 2026

**SleepFM a déjà vu des échelles très différentes d'une nuit à l'autre.** Sur les 360 nuits
d'apprentissage de la sonde respiratoire, l'échelle des sangles varie d'un facteur 12 à 13
entre les percentiles 5 et 95. Pour le débit, ce facteur atteint environ 800. D'après la
lecture fixée dans le plan (P95/P5 > 5), un alignement simple sur la médiane PhysioNet suffit
probablement pour NSRR. La décision reste au tuteur ; aucun résultat NSRR n'existe.

## Protocole

[Plan fixé avant calcul](../plans/Plan%20amplitude%20sangles%20d%C3%A9bit%20PhysioNet%2013-11.md)
(commit `37ac070`, `code_modified: false`). Valeurs brutes du .mat, telles que SleepFM les
reçoit, sur tout l'enregistrement. Échelle robuste = 1,4826 × écart absolu médian, par nuit.
Aucun modèle, aucun label, aucune nuit de validation ou de test.

## Échelle robuste par nuit (360 nuits)

| Canal | P5 | Q1 | Médiane | Q3 | P95 | P95 / P5 |
|---|---|---|---|---|---|---|
| ABD | 56 | 107 | **173** | 270 | 752 | 13 |
| CHEST | 36 | 59 | **83** | 144 | 412 | 12 |
| AIRFLOW | 1,5 | 5,9 | **13** | 86 | 1 189 | 802 |
| SaO2 (témoin) | 0 | 961 | 962 | 964 | 1 924 | — |

- **Débit** : les valeurs s'étalent sur trois ordres de grandeur, sans former deux groupes
  nets. Les séries d'enregistrements tr03, tr04 et tr05 ont des médianes proches (9, 12 et
  15). Une nuit a un débit constant.
- **Sangles** : les médianes sont proches d'une série à l'autre (ABD : 148 à 168 ; CHEST :
  77 à 79).
- **SaO2** : l'échelle de 962 correspond à environ 1 point de saturation (655,35 unités par %),
  ce qui confirme la conversion prévue. Dans 21 nuits, plus de la moitié des échantillons de SaO2 ont la même valeur
  (échelle robuste nulle).
- **Résolution** : environ 29 % (ABD), 42 % (CHEST) et 14 % (AIRFLOW) des échantillons
  successifs sont identiques (médiane par nuit). Le signal est donc peu résolu ou
  suréchantillonné. C'est à garder en tête face aux sangles NSRR à 10 ou 32 Hz.

## Proposition au tuteur (inchangée par ce calcul)

Pour chaque nuit NSRR et chaque canal (ABD, CHEST, AIRFLOW), multiplier le signal pour que son
échelle robuste égale la médiane PhysioNet : **173 (ABD), 83 (CHEST), 13 (AIRFLOW)**. La forte
variation déjà vue par SleepFM rend la règle peu sensible au choix exact de la référence. OSF
n'est pas concerné, car il normalise déjà chaque canal sur la nuit. Un canal NSRR d'échelle robuste nulle serait
refusé plutôt que multiplié.

## Limites

Statistiques sur tout l'enregistrement, éveil compris. L'unité nominale des sangles et du débit
(µV) n'a pas de sens physique : seule l'échelle relative compte. La tolérance de SleepFM à un
changement d'échelle n'est pas mesurée ici : elle se verra sur la confirmation NSRR.

## Fichiers

- [Résultats complets](../../results/respiratory_amplitude_physionet2018.json)
  (SHA-256 `a42cd399…99bc`)
- Script : `scripts/respiratory_amplitude_physionet2018.py`
