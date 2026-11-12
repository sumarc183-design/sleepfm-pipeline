# SleepFM / OSF — plan des séquences respiratoires, 12 novembre 2026

Complément descriptif sur les 100 nuits de test déjà explorées. Réutiliser les
probabilités archivées et les huit configurations de seuils déjà évaluées
(F1 initial et cibles de validation 70/80/90 %, pour chaque modèle). Aucun
nouveau seuil, entraînement, téléchargement ou usage de NSRR et des 494 sujets
de confirmation.

## Définitions fixées avant calcul

- Une séquence regroupe des fenêtres sélectionnées consécutives : même nuit,
  indices d'époque consécutifs et débuts espacés de 30 s, tolérance absolue
  0,01 s pour les décalages numériques existants. Toute fenêtre non sélectionnée,
  exclusion, lacune ou changement de nuit coupe la séquence. Aucun comblement.
- Séquences prédites positives : fenêtres dont le score atteint le seuil.
  Les classer en entièrement négatives dans les annotations, entièrement
  positives ou mixtes. Une séquence mixte contient des vrais et faux positifs.
- Séquences de faux positifs : regroupement séparé des seules fenêtres
  prédites positives et annotées négatives. Elles peuvent être incluses dans
  une séquence prédite mixte : ne pas confondre les deux décomptes.
- Durée nominale : nombre de fenêtres × 30 s. Conserver aussi les bornes
  temporelles observées ; ne pas inclure les lacunes dans cette durée.
- Rapporter effectifs, fraction des séquences de faux positifs isolées
  (une fenêtre), durée médiane, 90e percentile et maximum. Conserver les
  intervalles de toutes les configurations, sans sélection de nuits favorables.
- Charge par nuit : séquences de faux positifs, minutes de faux positifs et
  exposition évaluée. Exposition = nombre de fenêtres notées × 30 s ; ce
  n'est ni le temps de sommeil ni toute la durée d'enregistrement.
- Taux global = somme des séquences de faux positifs / somme des heures
  évaluées. IC à 95 % par bootstrap des 100 nuits, 500 tirages, graine 42,
  avec les seuils fixes. Décrire aussi la médiane et l'IQR des comptes par nuit.

## Contrôles et limites

Vérifier empreintes des références et prédictions, nuits, indices, temps et
probabilités. Retrouver pour chacune des huit configurations les matrices,
fenêtres faussement positives et fenêtres positives prédites de l'analyse
des seuils. Les séquences doivent couvrir exactement les fenêtres sélectionnées,
une seule fois chacune, et ne jamais franchir une lacune.

Ce regroupement n'est pas un détecteur d'événements : aucun appariement aux
événements respiratoires complets, aucune sensibilité événementielle ni fausse
alerte clinique n'est mesurée. Un épisode mixte n'est pas un épisode validé
cliniquement. Le taux dépend de la règle de regroupement et des fenêtres exclues.
Toute future règle d'alerte nécessitera un choix sur validation, puis une
évaluation indépendante.

Plan et compte rendu dans `Journal`, artefacts de calcul dans le projet.
