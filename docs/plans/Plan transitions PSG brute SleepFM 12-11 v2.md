# Transitions sur PSG brute — plan v2 du 12 novembre 2026

La version initiale a été arrêtée avant tout nouveau score : sur tr03-0052,
84 labels uniformes différaient, avec un écart de durée de 0,005 s à certaines
frontières entre les marqueurs WFDB et les vecteurs HDF5. Cette version reprend
la source exacte de l’évaluation uniforme. Aucun score n’a servi à ce changement.

Complément exploratoire sur les 100 nuits de test déjà ouvertes. Aucun
entraînement, réglage ou changement des deux modèles figés de stadification.
Ni données NSRR ni 494 sujets de confirmation supplémentaires.

## Référence fixée avant les scores

- Réutiliser toutes les fenêtres fixes de 30 s et les embeddings de
  `embeddings/physionet2018_raw_grid/`. Les modèles voient toute la nuit.
- Calculer les durées depuis les indicateurs HDF5 `-arousal.mat` du jeu
  officiel, exactement comme l’évaluation uniforme. Télécharger seulement
  ces petites annotations si elles manquent, aucun signal PSG. Vérifier les
  six colonnes, le codage one-hot et les longueurs de l’en-tête.
- **Contrôle obligatoire :** les fenêtres entièrement d'un stade doivent
  retrouver exactement les labels `y_grid` du run précédent, sur les 100 nuits.
  En cas d'écart, arrêter avant toute publication de nouveaux scores.
- Sur une fenêtre entièrement annotée : attribuer le stade qui couvre
  **strictement plus de 15 secondes**. Une égalité, ou l'absence de majorité
  absolue, reste non notée. Toute présence d'indéfini exclut la fenêtre.
- Séparer les fenêtres uniformes, les fenêtres à un seul échantillon
  minoritaire, les transitions à majorité définie avec plus d’un échantillon
  minoritaire, les transitions sans majorité, l’indéfini complet et l’indéfini
  partiel. La somme des six groupes doit retrouver toutes les fenêtres du cache.
  Rapporter séparément les cas d’un échantillon, sans en déduire une transition
  clinique réelle ni une erreur d’annotation.

Cette règle de majorité est choisie pour l'évaluation ; elle ne remplace pas
une cotation clinique de la fenêtre. Les fenêtres mixtes restent difficiles
à interpréter même lorsqu'une majorité existe.

## Mesures et vérifications

- Baseline et BiGRU figés du run à 500 sujets : vérifier leurs empreintes et
  contrats, prédire sans entraînement ni suppression du contexte de la nuit.
- κ, F1 macro, accuracy, matrice et F1 par stade : séparément sur fenêtres
  uniformes, mixtes d’un seul échantillon, transitions plus substantielles et
  ensemble notable sous la nouvelle règle.
- IC à 95 % par bootstrap de sujets, 500 tirages, graine 42. Rapporter les
  classes et effectifs de chaque groupe ; un groupe vide est non défini.
- Reproduire les κ et F1 publiés sur les fenêtres uniformes avant de publier
  le complément. Les scores uniformes et transitions portent sur des fenêtres
  différentes : leur écart ne mesure pas l'effet d'un changement de modèle.
- Conserver probabilités, labels, groupes, empreintes des entrées, modèles,
  code et présent plan. Ne pas écraser le résultat de la grille uniforme.

Compte rendu daté dans `Journal`, artefacts de calcul dans `sleepfm-pipeline/results/`.
