# SleepFM / OSF — séquences respiratoires, plan v2 du 12 novembre 2026

Le plan initial est conservé. Son premier contrôle a arrêté le calcul avant
tout décompte : les lignes des archives suivent les noms de fichiers, pas
l'ordre chronologique. Cette version explicite la remise en ordre avant
l'analyse ; les définitions de séquences, seuils, durées et IC restent celles
du plan initial. Aucun résultat de séquences n'a servi à cette précision.

## Ordre des données

- Vérifier les empreintes, dimensions et identités comme prévu initialement.
- Construire une permutation déterministe par sujet puis indice d'époque
  entier ; appliquer exactement cette permutation à tous les vecteurs,
  labels et probabilités. Archiver la correspondance avec les lignes d'origine.
- Refuser doublons, temps non croissants ou chevauchements supérieurs à la
  tolérance de 0,01 s après tri. Toute lacune coupe les séquences.
- Ne modifier aucune archive source. Retrouver les matrices et comptes de
  fenêtres des huit configurations, indépendants de leur ordre de stockage.

## Définitions et mesures conservées

100 nuits déjà explorées ; mêmes modèles et huit seuils précédents, aucun
entraînement, téléchargement, nouveau seuil, NSRR ou sujet de confirmation.

Une séquence exige des fenêtres sélectionnées consécutives, de la même nuit,
à indices consécutifs et débuts espacés de 30 s à 0,01 s près. Une fenêtre
non sélectionnée ou une lacune coupe ; aucun comblement. Durée nominale :
nombre de fenêtres × 30 s, avec bornes observées conservées séparément.

Compter séparément les séquences prédites positives (entièrement négatives,
entièrement positives ou mixtes dans les annotations) et les séquences des
seules fenêtres faussement positives. Ces dernières peuvent appartenir à
une séquence prédite mixte. Retrouver exactement toutes les fenêtres sélectionnées.

Rapporter effectifs, fraction des faux positifs regroupés en une seule fenêtre,
durée médiane, 90e percentile, maximum et comptes par nuit (médiane, IQR).
Exposition = nombre de fenêtres évaluées × 30 s, sans la confondre avec le
temps de sommeil ou d'enregistrement. Taux global de séquences de faux positifs
par heure évaluée, IC à 95 % par bootstrap des 100 nuits, 500 tirages, graine 42,
seuils fixes. Conserver tous les intervalles, sans choisir des nuits favorables.

Ce regroupement n'est pas une évaluation événementielle ou d'alertes cliniques.
Aucun appariement aux événements complets ni sensibilité événementielle.
Les nuits sont déjà explorées et les IC ne couvrent pas le choix des seuils.
Toute future règle d'alerte devra être choisie sur validation puis confirmée
indépendamment. Plan et compte rendu dans `Journal`, artefacts dans le projet.
