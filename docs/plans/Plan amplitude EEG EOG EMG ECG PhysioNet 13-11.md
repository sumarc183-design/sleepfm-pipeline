# Plan — amplitude des voies EEG, EOG, EMG et ECG sur PhysioNet 2018 (13 novembre 2026)

**Fixé avant calcul.** Calcul descriptif, sans label, sans modèle, sans nuit de validation ou de test.

## Pourquoi

Les EEG, EOG, EMG et ECG d'UCDDB ne sont pas calibrés (en-tête : 0 à 1 pour toute la plage
numérique). Pour SleepFM, la règle validée pour les sangles est étendue à ces voies : l'échelle
robuste de chaque nuit est ramenée à la médiane PhysioNet. Il faut donc ces médianes
([plan du contrôle externe](Plan%20contr%C3%B4le%20externe%20IAH%20UCDDB%20PSG-Audio%2013-11.md),
décision 4).

## Protocole

- **Nuits** : les 360 nuits d'apprentissage de la sonde respiratoire (même liste que le calcul
  d'amplitude des sangles du 13 novembre).
- **Voies** : `C3-M2`, `C4-M1`, `E1-M2`, `Chin1-Chin2`, `ECG` ; valeurs brutes du .mat, telles
  que SleepFM les reçoit, sur tout l'enregistrement.
- **Mesure** : échelle robuste par nuit = 1,4826 × écart absolu médian ; médiane sur les nuits.
- **Script** : `scripts/respiratory_amplitude_physionet2018.py --channels … --out
  results/eeg_amplitude_physionet2018.json --plan <ce fichier>`, code commité avant le lancement.

## Usage prévu

Les médianes deviennent les cibles de la règle d'échelle de SleepFM pour les voies non calibrées
d'UCDDB. Aucune autre décision n'en dépend. Une voie dont la médiane serait nulle n'est pas utilisée.
