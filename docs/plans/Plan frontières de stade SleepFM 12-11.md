# SleepFM — plan d'analyse des frontières de stade, 12 novembre 2026

Analyse descriptive complémentaire, fixée avant ce calcul, sur les mêmes
100 nuits déjà examinées. Les scores des fenêtres uniformes et des fenêtres
à un échantillon minoritaire sont déjà connus. Aucun entraînement, nouveau
seuil, choix de modèle ou utilisation des sujets de confirmation et de NSRR.

## Questions et méthode

- Reprendre les prédictions archivées de `raw_grid_transitions_physionet2018`,
  sans recalculer les modèles. Vérifier les empreintes des prédictions et de
  l'inventaire des annotations, puis celles des HDF5 et en-têtes des 100 nuits.
- Reconstruire les stades échantillon par échantillon et exiger les mêmes
  labels majoritaires et catégories que l'évaluation précédente, sur toutes
  les fenêtres. Arrêter si une divergence apparaît.
- Pour chaque fenêtre à un échantillon minoritaire : identifier ce stade,
  la position de cet échantillon (premier, dernier ou intérieur) et les
  stades majoritaires des fenêtres voisines lorsqu'ils sont définis.
- Décrire la fréquence par stade et par paire majorité/minorité. Conserver
  toutes les paires dans le résultat ; limiter les interprétations détaillées
  aux paires d'au moins 100 fenêtres et présentes dans au moins 10 nuits.
- Pour chaque sonde figée, compter les prédictions du stade majoritaire,
  du stade minoritaire et d'un autre stade. Refaire les matrices globales
  pour vérifier l'accord avec le résultat précédent.
- Donner les proportions de ces trois issues sur le groupe entier et leurs
  IC à 95 % par bootstrap des 100 nuits, 500 tirages, graine 42. Les tableaux
  par paire et par stade restent descriptifs, sans test de supériorité.

## Limites et suite

Un échantillon minoritaire ne prouve pas une transition clinique réelle ni une
erreur de cotation. Prédire le stade minoritaire n'est pas une preuve de retard
du modèle. Sans signaux ni examen médical, cette analyse décrit les annotations
et les erreurs ; elle n'en établit pas la cause.

Le cache OSF disponible utilise la grille annotée, légèrement décalée : il
ne sera pas présenté comme une évaluation OSF de la grille brute. Celle-ci
nécessiterait une extraction distincte.

Compte rendu et suggestions dans `Journal`, fichiers de calcul dans le projet.
