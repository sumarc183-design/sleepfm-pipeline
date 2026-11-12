# Première baseline respiratoire — plan du 12 novembre 2026

Exploration sur PhysioNet 2018, indépendante de la demande NSRR. Ce plan est
écrit avant le calcul des scores de cette nouvelle tâche ; il ne constitue pas
un préenregistrement externe. Les sujets ont déjà été examinés pour les stades.

## Question et cible

Une logistique sur les embeddings SleepFM figés peut-elle reconnaître les
fenêtres contenant des **apnées (obstructives, centrales, mixtes) ou hypopnées** ?
La cible est l'union des annotations WFDB `resp_obstructiveapnea`,
`resp_centralapnea`, `resp_mixedapnea`, `resp_hypopnea` de l'extension `.arousal`.
Le vecteur d'arousals du challenge n'est pas un label d'apnée.

- Réutiliser les fenêtres annotées SleepFM de 30 s et leurs temps d'origine.
- Positif : au moins **10 secondes** couvertes par l'union des événements cibles.
- Négatif : aucune seconde couverte. Entre 0 et 10 s : exclu, effectif publié.
- Fusionner les intervalles qui se chevauchent pour éviter un double comptage.
- Arrondir les débuts de fenêtre à l'échantillon d'origine le plus proche.

Ce n'est ni une mesure d'IAH, ni un diagnostic d'apnée obstructive, ni un
détecteur d'événements à la seconde. Les hypopnées ne sont pas toutes attribuées
à une cause obstructive. Les résultats ne simulent pas un téléphone ou une montre.

## Découpage et réglages fixés

- Même split de **400 sujets train / 100 test** ; ne pas utiliser les 494 sujets
  de confirmation des stades pour développer cette baseline.
- Parmi les 400 : 40 sujets de validation tirés avec la graine 42 ; les 360
  autres servent à ajuster la logistique. Aucun sujet partagé.
- Embeddings complets de 1 536 dimensions ; pas de nouvelle extraction GPU.
- `LogisticRegression(C=1, class_weight="balanced", max_iter=1000)` ; aucun
  balayage. Pas de réentraînement après sélection du seuil.
- Seuil maximisant le F1 sur la validation seulement ; en cas d'égalité,
  prendre le plus grand seuil. Garder le modèle et ce seuil pour le test.
- Critère principal descriptif : **average precision (AP)** sur le test, avec
  prévalence comme référence constante. Secondaires : ROC-AUC, F1, précision,
  sensibilité, spécificité, matrice de confusion et nombre de nuits/fenêtres.
- IC à 95 % sur AP et AP moins prévalence : 500 tirages de sujets avec remise,
  graine 42. Rapporter les tirages sans les deux classes, s'il y en a.
- Publier le résultat même décevant, avec exclusions, empreintes, paramètres,
  sujets effectivement utilisés, probabilités et limites de convergence.

## Démonstration

Une page HTML locale, sans serveur ni ressource externe : hypnogramme de
référence et prédictions de stades, probabilités et désaccords ; probabilité
respiratoire et événements annotés si disponibles. Une nuit de test choisie
par son identifiant, sans choisir le cas le plus favorable. Les probabilités
pondérées ne sont pas présentées comme un risque calibré.

## Sources

- [PhysioNet Challenge 2018, annotations et formats](https://physionet.org/content/challenge-2018/1.0.0/).
- Découpage : `configs/split_physionet2018.json` ; embeddings :
  `embeddings/physionet2018_v2/`. Empreintes à conserver avec le run.

## Exécution et résultat

Ce plan reste conservé tel quel après le run ; son empreinte est enregistrée
dans le résultat. Les commandes et le bilan figurent dans le guide et dans
`docs/RESPIRATORY_RESULTATS.md`.
