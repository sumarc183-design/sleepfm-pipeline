# Transitions sur PSG brute — plan du 12 novembre 2026

Complément exploratoire sur les 100 nuits de test déjà ouvertes. Aucun
entraînement, réglage ou changement des deux modèles figés de stadification.
Ni données NSRR ni 494 sujets de confirmation supplémentaires.

## Référence fixée avant les scores

- Réutiliser toutes les fenêtres fixes de 30 s et les embeddings de
  `embeddings/physionet2018_raw_grid/`. Les modèles voient toute la nuit.
- Reconstruire les durées par stade depuis les marqueurs WFDB W/N1/N2/N3/R
  des annotations déjà téléchargées. Avant le premier marqueur : indéfini.
  Intervalles demi-ouverts, référence bornée par la longueur de l'en-tête.
- **Contrôle obligatoire :** les fenêtres entièrement d'un stade doivent
  retrouver exactement les labels `y_grid` du run précédent, sur les 100 nuits.
  En cas d'écart, arrêter avant toute publication de nouveaux scores.
- Sur une fenêtre entièrement annotée : attribuer le stade qui couvre
  **strictement plus de 15 secondes**. Une égalité, ou l'absence de majorité
  absolue, reste non notée. Toute présence d'indéfini exclut la fenêtre.
- Séparer les fenêtres uniformes, les transitions à majorité définie, les
  transitions sans majorité, l'indéfini complet et l'indéfini partiel.
  La somme des cinq groupes doit retrouver toutes les fenêtres du cache.

Cette règle de majorité est choisie pour l'évaluation ; elle ne remplace pas
une cotation clinique de la fenêtre. Les fenêtres mixtes restent difficiles
à interpréter même lorsqu'une majorité existe.

## Mesures et vérifications

- Baseline et BiGRU figés du run à 500 sujets : vérifier leurs empreintes et
  contrats, prédire sans entraînement ni suppression du contexte de la nuit.
- κ, F1 macro, accuracy, matrice et F1 par stade : séparément sur fenêtres
  uniformes, transitions notables et ensemble notable sous la nouvelle règle.
- IC à 95 % par bootstrap de sujets, 500 tirages, graine 42. Rapporter les
  classes et effectifs de chaque groupe ; un groupe vide est non défini.
- Reproduire les κ et F1 publiés sur les fenêtres uniformes avant de publier
  le complément. Les scores uniformes et transitions portent sur des fenêtres
  différentes : leur écart ne mesure pas l'effet d'un changement de modèle.
- Conserver probabilités, labels, groupes, empreintes des entrées, modèles,
  code et présent plan. Ne pas écraser le résultat de la grille uniforme.

Compte rendu daté dans `Journal`, artefacts de calcul dans `sleepfm-pipeline/results/`.
