# SleepFM / OSF — stabilité des seuils, plan du 12 novembre 2026

Diagnostic sur les mêmes **40 nuits de validation** et leurs probabilités déjà
archivées. Aucun entraînement, téléchargement ou lecture de prédictions de test,
de NSRR ou des 494 sujets de confirmation. Les seuils des modèles existants sont
conservés ; les seuils recalculés ci-dessous servent uniquement au diagnostic.

## Règles fixées avant calcul

- Reprendre les deux scores et les quatre règles existantes : F1 maximal
  (plus grand seuil en cas d'égalité), cibles de sensibilité 70/80/90 %
  (plus grand seuil atteignant la cible, ex æquo inclus).
- Vérifier les empreintes de l'archive de validation, ses sujets et identités.
  Retrouver les huit seuils et matrices publiés avec les 40 nuits complètes.

## Bootstrap incluant la sélection

- **500 tirages, graine 42** : tirer 40 nuits avec remise. Chaque occurrence
  d'une nuit compte toutes ses fenêtres ; appliquer ses multiplicités comme
  poids lors de la sélection du seuil. Même tirage pour les deux modèles et
  les quatre règles. Aucune permutation de fenêtres entre nuits.
- Recalculer le seuil dans chaque échantillon de calibration, puis l'appliquer
  aux nuits absentes de ce tirage (**hors échantillon / OOB**), chacune une
  seule fois et sans pondération par les multiplicités de calibration.
- Écarter et compter les tirages sans nuit OOB ou sans les deux classes dans
  la calibration ou l'OOB ; ne pas remplacer ces tirages.
- Conserver seuils, matrices, multiplicités, raisons d'exclusion et nombres de
  nuits OOB. Rapporter médiane et percentiles 2,5/97,5 % des seuils et des
  sensibilités, spécificités, précisions et F1 OOB. Pour les cibles, décrire la
  fraction de tirages OOB sous la cible ; ce n'est pas une probabilité de risque.
- Les percentiles OOB décrivent une distribution de rééchantillonnage, pas un
  IC clinique de généralisation. Les échantillons OOB sont petits et variables ;
  les tirages sont dépendants et la calibration contient des nuits répétées.

## Contrôle complémentaire : laisser une nuit de côté

Pour chacune des 40 nuits, choisir les seuils sur les 39 autres et prédire la
nuit laissée de côté. Chaque fenêtre reçoit exactement une prédiction ainsi
obtenue. Rapporter les métriques agrégées et les 40 seuils par règle/modèle.
Ce sont des résultats de validation croisée du **choix du seuil**, avec un
modèle déjà figé ; ils ne décrivent pas un unique seuil utilisable tel quel.
Ne pas leur attribuer un IC à seuil fixe ni une confirmation indépendante.

## Portée et conservation

Les 40 nuits ont déjà servi à d'autres analyses et restent du développement.
Cette étude quantifie l'incertitude de calibration au sein de cette cohorte,
pas l'effet du réentraînement, d'une nouvelle cohorte ou de nouveaux capteurs.
Elle ne choisit aucune règle d'alerte et ne corrige pas à elle seule le constat
de transfert validation/test. Celui-ci restera ouvert jusqu'à confirmation
sur des données indépendantes.

Plan et compte rendu en un seul exemplaire dans `Journal`, résumé hebdomadaire.
Artefacts, tirages, prédictions et archive du code/plan dans le projet ; aucun
résultat existant écrasé, aucun commit ni push prévu par ce plan.
