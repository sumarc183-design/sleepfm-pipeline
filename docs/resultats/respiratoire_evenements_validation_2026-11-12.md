# SleepFM / OSF — événements sur validation, 12 novembre 2026

**Le recouvrement d'un événement et son identification séparée donnent des
résultats très différents.** Pour OSF au seuil initial, les fenêtres positives
recouvrent **72,5 %** des événements évaluables ; l'appariement un à un aux
séquences en retrouve **33,5 %** sous la règle retenue.

Diagnostic sur les **40 nuits de validation**, 34 437 fenêtres. Ces nuits ont
déjà servi à choisir les seuils : résultats apparents de développement, sans
estimation indépendante de généralisation. Aucun nouveau calcul sur les
prédictions des 100 nuits de test, aucun entraînement, seuil modifié ou
téléchargement ; ni NSRR ni 494 sujets de confirmation utilisés.

## Quels événements sont réellement évaluables ?

| Couverture par les fenêtres disponibles | Événements |
|---|---:|
| Complète, à 0,01 s près | 3 111 |
| Partielle | 1 692 |
| Nulle | 202 |
| **Total annoté** | **5 005** |

Parmi les événements complètement couverts, 15 sont trop courts sous la règle
de durée minimale. Le dénominateur commun des deux rappels est donc **3 096
événements**, soit **61,9 % des annotations**. Les 1 909 autres événements
restent hors de ces rappels. Au total, 53 événements sont trop courts sous
la règle de 10 s à 0,01 s près, toutes catégories de couverture confondues.

| Type annoté | Total | Éligibles |
|---|---:|---:|
| Apnées obstructives | 1 979 | 1 197 |
| Apnées centrales | 764 | 471 |
| Apnées mixtes | 92 | 36 |
| Hypopnées | 2 170 | 1 392 |

La sélection des fenêtres limite donc la portée de l'évaluation. Les résultats
ci-dessous ne représentent pas tous les événements des nuits, et la faible
couverture des apnées mixtes mérite une attention particulière.

## Deux mesures sur le même dénominateur

- **Recouvrement** : au moins 10 s d'un événement sont recouvertes par l'union
  des fenêtres positives, avec tolérance de 0,01 s. Une longue séquence peut
  recouvrir plusieurs événements.
- **Appariement un à un** : chaque séquence et chaque événement participent à
  une seule paire au maximum. Une paire exige au moins 10 s de recouvrement
  réel par les fenêtres de la séquence, à 0,01 s près. Le calcul maximise le
  nombre de paires, puis la somme des IoU pour départager.

Les séquences utilisent le regroupement précédent, sans comblement des lacunes.
Les événements annotés restent individuels, y compris ceux qui se touchent.

| Modèle | Seuil existant | Recouvrement | Appariement un à un |
|---|---|---:|---:|
| SleepFM | F1 initial | 71,6 % | 29,0 % |
| SleepFM | Cible validation 70 % | 75,3 % | 30,0 % |
| SleepFM | Cible validation 80 % | 84,7 % | 32,9 % |
| SleepFM | Cible validation 90 % | 92,4 % | 34,8 % |
| OSF | F1 initial | 72,5 % | 33,5 % |
| OSF | Cible validation 70 % | 73,2 % | 33,7 % |
| OSF | Cible validation 80 % | 83,0 % | 36,3 % |
| OSF | Cible validation 90 % | 92,4 % | 37,6 % |

Exemple OSF initial : **2 244** événements recouverts et **1 037** appariés
sur 3 096. IC à 95 % : **[64,0 ; 78,6] %** pour le recouvrement et
**[26,8 ; 41,7] %** pour l'appariement. Bootstrap des 40 nuits, 500 tirages,
graine 42, seuils fixes ; l'incertitude du choix des seuils n'est pas incluse.
Tous les comptes et IC sont dans le JSON.

Le seuil plus bas améliore surtout le recouvrement dans ce diagnostic.
La localisation et la séparation des événements restent à développer et à
valider. Aucun test de supériorité entre modèles n'a été réalisé ici.

## Vérifications et limites

- [Plan fixé avant calcul](../plans/Plan%20%C3%A9v%C3%A9nements%20respiratoires%20validation%20SleepFM%20OSF%2012-11.md).
- Empreintes des sources, probabilités, en-têtes et annotations contrôlées.
  Les labels binaires reconstruits retrouvent l'archive des 40 nuits et les
  huit matrices de validation retrouvent l'étude des seuils.
- Vérification indépendante en échantillons entiers : couverture et éligibilité
  identiques ; écart maximal des durées **4,4 × 10⁻¹² s**. Un autre algorithme
  d'appariement biparti retrouve les cardinalités maximales des huit configurations.
  Paires uniques et recouvrements admissibles vérifiés ; IC reproduits sans écart.
- **75 tests ciblés réussis, dont 9 nouveaux**, couvrant notamment événements
  voisins, chevauchements, lacunes, compétition entre paires et lecture de la
  seule validation. Lint, formatage et contrôle des espaces du diff réussis.
  La suite globale et les tests lents n'ont pas été relancés.

Ce calcul n'est pas une validation clinique. Les règles d'appariement sont
exploratoires et les événements partiellement observés restent exclus. Il ne
fournit ni précision événementielle, ni faux événements par heure, ni IAH.

Révision de base : `2c61add`, avec nouveau code non commité signalé dans
l'environnement du résultat. Lecteur WFDB 4.1.2 utilisé dans un environnement
isolé ; dépendances principales inchangées. Code et plan exécutés archivés.

## Preuves conservées

- [Résultats, IC et comptes par nuit](../../results/respiratory_events_validation_physionet2018.json)
- [Références individuelles, couvertures et paires](../../results/respiratory_events_validation_physionet2018.events.npz)
- [Séquences et liens vers les événements](../../results/respiratory_events_validation_physionet2018.runs.npz)
- [Permutation vers les lignes de validation d'origine](../../results/respiratory_events_validation_physionet2018.ordering.npz)
- [Archive du code et du plan](../../results/respiratory_events_validation_physionet2018.code.zip)

## À faire

- Faire définir avec le tuteur et le médecin l'événement et l'alerte visés,
  ainsi que la règle d'appariement pertinente pour cet usage.
- Examiner sur validation les lacunes de couverture et la séparation des
  événements successifs avant de choisir une nouvelle méthode.
- Préparer ensuite une évaluation de nuits complètes, avec durée de sommeil
  et événements observables, avant toute estimation d'IAH ou confirmation
  sur une cohorte indépendante.
