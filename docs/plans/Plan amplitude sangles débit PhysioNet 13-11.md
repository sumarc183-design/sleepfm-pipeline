# Amplitude des sangles et du débit sur PhysioNet 2018 — plan du 13 novembre 2026

**Objet** : donner au tuteur des chiffres pour décider de la mise à l'échelle des sangles
(ABD, CHEST) et du débit (AIRFLOW) avant d'appliquer SleepFM sur NSRR
([ébauche du plan NSRR](Plan%20confirmation%20IAH%20NSRR%20SleepFM%20OSF%2013-11.md), point 2).
Analyse **descriptive** : aucun modèle, aucun label, aucune nuit de validation ou de test.

## Données

Les **360 nuits d'apprentissage** (`fit`) de la partition respiratoire
(`results/respiratory_baseline_physionet2018.json`). Signaux bruts du .mat, valeurs telles que
SleepFM les reçoit (1 unité par µV nominal, en-tête .hea), sur tout l'enregistrement.

## Mesures

Par nuit et par canal (ABD, CHEST, AIRFLOW ; SaO2 comme témoin) : médiane, échelle robuste
(1,4826 × écart absolu médian), écart-type, percentiles 1 et 99, part d'échantillons constants.
Entre les nuits : médiane, quartiles, percentiles 5 et 95 de l'échelle robuste, et rapport
P95/P5 (variation d'échelle que SleepFM a déjà vue à l'entraînement).

## Lecture prévue

- Variation entre nuits faible (P95/P5 < 2) : SleepFM a vu peu de variation d'échelle, ce qui
  rend l'alignement sur l'échelle PhysioNet nécessaire.
- Variation forte (P95/P5 > 5) : SleepFM a déjà vu des échelles très différentes ; un alignement
  simple sur la médiane PhysioNet suffit probablement.

Règle proposée au tuteur, inchangée par ce calcul : pour chaque nuit NSRR et chaque canal,
multiplier le signal pour que son échelle robuste égale la médiane PhysioNet mesurée ici.
La décision reste au tuteur ; aucun résultat NSRR n'existe à ce stade.

## Exécution

`scripts/respiratory_amplitude_physionet2018.py` : téléchargement sujet par sujet, fichiers
supprimés après lecture, sortie `results/respiratory_amplitude_physionet2018.json` (valeurs par
nuit, résumé, empreinte du plan et de la liste, environnement).
