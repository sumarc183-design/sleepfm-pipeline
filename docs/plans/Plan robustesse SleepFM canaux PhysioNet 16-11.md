# Plan — robustesse de SleepFM aux canaux absents et à la respiration sous-échantillonnée (16 novembre 2026)

**Fixé avant calcul. Diagnostic descriptif : aucune règle, aucun modèle, aucun seuil ne change.**

## Pourquoi

Sur UCDDB, SleepFM ne produit aucune fenêtre positive, même avec la règle d'échelle corrigée
([compte rendu](Compte%20rendu%20contr%C3%B4le%20externe%20UCDDB%2013-11.md)). Deux causes probables
reviendront sur NSRR : des voies EEG absentes (mises à zéro) et une respiration enregistrée à
basse fréquence. Ce test mesure, sur PhysioNet, ce que chaque dégradation coûte à SleepFM, pour
savoir avant le passage NSRR si SleepFM peut y être jugé.

## Protocole

- **Nuits** : les 40 nuits de validation respiratoires (`results/respiratory_ahi_physionet2018.nights.npz`,
  partition `validation`). Aucune nuit de test, aucune des 494 nuits réservées.
- **Préparation** : grille fixe de 30 s (`RawNightDataset`), comme pour NSRR. Dégradations
  appliquées à la nuit à 256 Hz :
  - `intact` : aucune ;
  - `eeg_mesa` : F3, F4, C3, O1, O2 mis à zéro (MESA garde C4, EOG gauche, menton) ;
  - `resp_8hz` : sangles, débit et SaO2 passés à 8 Hz puis remontés à 256 Hz (UCDDB) ;
  - `mesa_like` : `eeg_mesa` + sangles et débit à 32 Hz, SaO2 à 1 Hz ;
  - `shhs_like` : F3, F4, O1, O2 à zéro + sangles et débit à 10 Hz, SaO2 à 1 Hz ;
  - `ucddb_like` : F3, F4, O1, O2 à zéro + `resp_8hz`.
- **Modèle** : sonde SleepFM figée et son seuil (`results/respiratory_ahi_physionet2018.json`).
- **Mesures par condition** : taux de fenêtres positives par nuit ; AUC de ce taux pour IAH ≥ 15
  (IAH de référence de ces nuits) ; Spearman avec l'IAH de référence ; nuits à taux nul ;
  rapport du taux médian à celui d'`intact`.
- **Lecture fixée avant** : une condition est jugée **rédhibitoire** pour SleepFM si son AUC
  tombe sous 0,70 ou si plus de la moitié des nuits ont un taux nul. Sinon, la dégradation est
  rapportée telle quelle. Le passage NSRR n'est pas modifié par ce test : il sert à interpréter
  le résultat de SleepFM et à prévenir le tuteur.

## Limites

40 nuits, déjà utilisées pour la calibration de l'IAH. Ce test isole les canaux et la fréquence,
pas les autres différences entre appareils (capteurs, filtres, population).
