# SleepFM / OSF — événements sur validation, plan du 12 novembre 2026

Premier diagnostic de développement sur les **40 nuits de validation seulement**.
Les huit seuils existants sont conservés ; aucun seuil, filtre ou modèle choisi
à partir de ce calcul. Ne lire aucune nouvelle prédiction des 100 nuits de test,
aucun sujet de confirmation ou donnée NSRR. Aucun entraînement ou téléchargement.

## Références et couverture

- Reprendre les probabilités de validation archivées par l'étude des seuils.
  Vérifier empreintes, partitions, labels et identité des fenêtres. Trier tous
  les vecteurs ensemble par sujet et indice entier, sans changer les archives.
- Lire les annotations WFDB existantes et conserver chaque événement des quatre
  types respiratoires séparément, sans fusionner les événements voisins.
  Vérifier paires début/fin, fréquences et nombres d'événements contre le manifeste.
- Comme pour les labels initiaux : arrondir le début des fenêtres au plus proche
  échantillon de l'en-tête ; fenêtre de 30 s. Refaire les labels binaires sur
  les fenêtres évaluées et exiger leur identité avec l'archive de validation.
- Mesurer la durée de chaque événement couverte par l'union des fenêtres évaluées.
  Compter séparément couverture nulle, partielle et complète à **0,01 s près**.
  Ne pas combler les lacunes. La tolérance ne crée pas de nouvelle fenêtre.
- Le dénominateur des deux rappels ci-dessous est commun : événements complètement
  couverts à cette tolérance et durant au moins **10 s à 0,01 s près**. Rapporter
  les événements exclus, les durées trop courtes et les effectifs par type.

## Deux mesures distinctes, sans choix après résultats

1. **Recouvrement par fenêtres positives** : événement dont au moins 10 s
   (tolérance 0,01 s) sont couvertes par l'union des fenêtres prédites positives.
   Plusieurs événements peuvent être couverts par une même longue séquence ;
   cette mesure ne démontre pas leur localisation individuelle.
2. **Appariement un à un des séquences** : reprendre le regroupement temporel
   strict précédent (même nuit, indices consécutifs, espacement 30 s à 0,01 s
   près ; lacunes coupées). Chaque séquence peut correspondre au plus à un
   événement et réciproquement. Une paire exige au moins 10 s de recouvrement
   réel avec les fenêtres de la séquence (tolérance 0,01 s). Maximiser d'abord
   le nombre de paires, puis la somme des IoU pour départager. Conserver les
   correspondances et les durées, pas seulement les taux.

Rapporter ces deux mesures sur les huit configurations. Par type, décrire la
couverture et le premier rappel ; ne pas interpréter la répartition par type
des paires globales lorsque des appariements concurrents existent.

IC à 95 % par bootstrap des **40 nuits**, 500 tirages, graine 42, seuils fixes.
Ils ne couvrent ni le choix des seuils ni l'entraînement. Enregistrer les
comptes par nuit, événements, couvertures, paires, empreintes et code exécuté.
Retrouver les huit matrices par fenêtre de validation avant toute publication.

## Portée

Ces mêmes nuits ont servi à choisir les seuils : résultats apparents de
développement, sans estimation indépendante de généralisation. La sélection
des fenêtres rend une partie des événements non évaluable. L'appariement
dépend de règles exploratoires et n'est pas une validation clinique.
Ne calculer ni précision événementielle, ni faux événements par heure, ni IAH :
les événements partiellement observés et les fenêtres manquantes l'empêchent ici.

Plan et compte rendu en un seul exemplaire dans `Journal`, résumé hebdomadaire
avec liens ; artefacts dans le projet. Le script accepte le plan par `--plan`
et l'archive avec le code pour préserver la relance depuis une autre machine.
