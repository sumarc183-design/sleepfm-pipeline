# Analyse des erreurs respiratoires — plan du 12 novembre 2026

Analyse descriptive des prédictions de test déjà sauvegardées. Aucun nouvel
entraînement, réglage de seuil, choix de modèle ou accès NSRR. Les 494 sujets
de confirmation des stades restent hors de cette analyse.

## Questions

1. Quelle proportion des fenêtres positives est manquée pour chaque type
   d'événement : apnée obstructive, centrale, mixte, hypopnée ?
2. Les erreurs se répartissent-elles différemment selon le stade expert ?
3. Quelle variabilité observe-t-on entre les 100 nuits de test ?

## Règles fixées avant le regroupement des résultats

- Conserver exactement les fenêtres et labels de la comparaison respiratoire :
  85 230 fenêtres retenues ; ne pas réintégrer les 3 902 fenêtres exclues.
- Réutiliser les seuils de validation déjà publiés, propres à chaque modèle.
- Vérifier les empreintes des prédictions, labels, caches et annotations.
  Reconstituer les labels à partir des annotations comme contrôle d'alignement.
- Répartir les fenêtres positives en groupes **exclusifs** : un seul type
  présent, ou « plusieurs types » dès que plusieurs types couvrent la fenêtre,
  même si l'un d'eux ne couvre qu'une petite partie. Ne pas compter une fenêtre
  dans plusieurs lignes. Les groupes doivent totaliser les 11 438 positives.
- Par type : effectif, nuits concernées, vrais positifs, faux négatifs et rappel
  pour les deux modèles. Pas d'AP ni de spécificité sur un groupe uniquement positif.
- Par stade expert et par nuit : effectifs, matrice de confusion, rappel,
  spécificité, précision, F1 et AP si les deux classes sont présentes.
- IC descriptifs du rappel par type, et de l'écart OSF − SleepFM : 1 000 tirages
  appariés des 100 sujets avec remise, graine 42. Ignorer les tirages sans
  positif du groupe et en rapporter le nombre. Aucun ajustement de multiplicité.
- Lister les cinq nuits au rappel OSF le plus bas parmi celles ayant au moins
  10 fenêtres positives, avec départage par identifiant. Ce choix sert à examiner
  les erreurs, pas à représenter la performance globale ou choisir une belle démo.

## Portée

La sensibilité est mesurée par fenêtre, pas par événement complet ou patient.
Les sous-groupes décrivent des associations, pas la cause d'une erreur. Aucun
désaccord ne démontre une faute d'annotation. Les différences de stade,
d'événement et de patient peuvent se confondre. Les petits groupes doivent
être signalés ; les IC ne constituent pas de nouvelles confirmations.

Résultat daté dans `Journal` ; tableaux structurés et labels de groupe dans
`sleepfm-pipeline/results/`. Le présent plan sera conservé sans modification.
