# Confirmation de l'IAH ≥ 15 sur NSRR — ébauche du 13 novembre 2026

**Ébauche, à valider avec le tuteur.** Elle est écrite avant tout accès aux données NSRR
(demande n° 30917 en attente) pour que la confirmation puisse être lancée dès l'ouverture.
Décision du tuteur (12 novembre) : la confirmation se fait sur NSRR, pas sur les 494 sujets.

## Règle figée (aucune modification après l'accès)

Issue du [résultat du 12 novembre](../resultats/respiratoire_iah_2026-11-12.md)
(`results/respiratory_ahi_physionet2018.json`, SHA-256 `eda69eb9…2990`) :

| | SleepFM | OSF |
|---|---|---|
| Modèle respiratoire | `respiratory_baseline_physionet2018.joblib` | `respiratory_osf_vs_sleepfm_physionet2018.joblib` |
| Seuil par fenêtre | 0,6934 | 0,7636 |
| Calibration `IAH = a + b·x` | a = 1,992 ; b = 0,869 | a = −1,898 ; b = 1,025 |
| Nuit signalée si | IAH estimé ≥ 15 | IAH estimé ≥ 15 |

`x` = fenêtres de 30 s prédites positives par heure de fenêtres disponibles. Aucun
réentraînement, aucun nouveau seuil, aucune recalibration sur NSRR.

## Données

- **Cohorte** : MESA d'abord (cohorte indépendante, adultes), SHHS ensuite si accordée.
  À trancher : une seule visite par participant (SHHS-1 seulement).
- **Sélection** : toutes les nuits avec annotations XML NSRR et au moins 1 h de sommeil.
  Liste figée et empreintée **avant** toute prédiction ; nuits écartées publiées avec leur motif.
- **Annotations** : `src/data/nsrr_annotations.py` (stades R&K regroupés en AASM, 4 types
  d'événements). Un libellé inconnu arrête la lecture : à vérifier sur les premiers fichiers.

## Référence

- **Principale** : même règle que PhysioNet 2018 et l'étude de faisabilité, tous les
  événements annotés des 4 types par heure de sommeil expert.
- **Secondaire** : IAH officiel NSRR à 3 % et 4 % de désaturation (variables du dictionnaire,
  par exemple `ahi_a0h3` et `ahi_a0h4`, **noms à vérifier**). La règle d'hypopnée de PhysioNet
  2018 n'étant pas documentée, l'écart entre les deux références sera rapporté.

## Évaluation (un seul passage)

- **Critère principal** : sensibilité et spécificité à IAH ≥ 15 pour chaque modèle, IC 95 %
  par bootstrap des participants (1 000 tirages, graine 42).
- **Secondaires** : AUC de `x` et écart apparié OSF − SleepFM ; erreur absolue moyenne,
  Spearman, Bland-Altman, κ pondéré linéaire sur 4 classes ; mêmes calculs avec la référence
  officielle NSRR.
- **Critère de réussite** : à fixer avec le tuteur avant l'accès. Proposition : la borne
  basse de l'IC de la sensibilité et celle de la spécificité au-dessus de 60 %.

## Points ouverts avant l'accès

1. **Canaux** : SleepFM a été appris sur 13 canaux de PhysioNet 2018, OSF sur 12 canaux.
   MESA et SHHS n'ont pas le même montage (moins d'EEG, capteurs de débit différents). Table
   de correspondance canal par canal à écrire et faire valider ; un canal absent est rempli
   de zéros comme pour OSF sur PhysioNet, et signalé.
2. **Fréquences et prétraitement** : rééchantillonnage vers 256 Hz (SleepFM) et 64 Hz (OSF)
   avec le même code que PhysioNet ; à tester sur des signaux synthétiques.
3. **Calcul** : extraction des embeddings sur Kaggle (GPU) ; volume MESA à estimer.
4. **Différences de population** : MESA à domicile, adultes plus âgés ; la calibration peut
   se déplacer. C'est précisément ce que la confirmation mesure : aucun ajustement.

## Limites connues

Règle développée sur 140 nuits d'une seule cohorte ; IAH estimé imprécis (limites de
Bland-Altman au-delà de ±20 /h sur PhysioNet). Changement de capteurs et de population.
