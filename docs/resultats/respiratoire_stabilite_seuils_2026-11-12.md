# SleepFM / OSF — stabilité des seuils, 12 novembre 2026

**Le choix du seuil varie nettement selon les nuits de calibration.** Pour
OSF à la cible de 80 %, la plage centrale du seuil bootstrap est **[0,478 ;
0,698]** ; le rappel des nuits absentes du tirage s'étend de **63,8 à 89,5 %**
entre les percentiles 2,5 et 97,5. La validation croisée laissant une nuit de
côté retrouve environ **80,0 %** de rappel agrégé.

Ce diagnostic quantifie l'incertitude de calibration sur les **40 nuits de
validation, 34 437 fenêtres**. Le constat de transfert de l'audit reste ouvert :
la généralisation demande une confirmation indépendante. Aucun nouveau calcul
sur les prédictions de test, entraînement, téléchargement ou changement des
modèles et seuils d'origine ; ni NSRR ni sujets de confirmation utilisés.

## Méthode

[Plan fixé avant calcul](../plans/Plan%20stabilit%C3%A9%20seuils%20respiratoires%20SleepFM%20OSF%2012-11.md).

- **500 tirages de 40 nuits avec remise**, graine 42. Les nuits répétées
  pondèrent toutes leurs fenêtres lors de la sélection du seuil. Même tirage
  pour les deux modèles et les quatre règles.
- Chaque seuil est évalué sur les seules nuits **absentes du tirage (OOB)**,
  chacune une seule fois. Les 500 tirages sont évaluables, sans exclusion.
- Médiane de **26 nuits distinctes** en calibration et **14 nuits OOB** ;
  plage centrale des groupes OOB : **10 à 19 nuits**. Ces groupes sont petits
  et les tirages dépendants.
- Contrôle complémentaire : choisir les seuils sur 39 nuits, prédire la
  quarantième, puis recommencer. Chaque fenêtre reçoit exactement une
  prédiction ainsi obtenue. Les modèles sont figés dans les deux méthodes.

La règle F1 est réappliquée à chaque calibration. Les cibles 70/80/90 % restent
des objectifs de sensibilité sur les nuits utilisées pour choisir le seuil.
Les métriques agrègent les fenêtres ; le rééchantillonnage porte sur les nuits.

## Distribution bootstrap avec sélection du seuil

Les crochets donnent les **percentiles 2,5 et 97,5 %**, pas des IC cliniques
de généralisation. La dispersion OOB comprend à la fois les changements de
seuil et la composition des petits groupes de nuits évaluées.

| Modèle | Règle | Seuil médian [plage centrale] | Rappel OOB médian [plage centrale] | Tirages OOB sous la cible |
|---|---|---|---|---:|
| SleepFM | F1 maximal | 0,694 [0,651 ; 0,732] | 65,1 % [46,4 ; 78,2] | — |
| SleepFM | Cible 70 % | 0,635 [0,513 ; 0,754] | 70,2 % [43,7 ; 85,6] | 49,6 % |
| SleepFM | Cible 80 % | 0,488 [0,374 ; 0,613] | 80,1 % [58,4 ; 91,8] | 49,6 % |
| SleepFM | Cible 90 % | 0,304 [0,224 ; 0,413] | 89,8 % [75,0 ; 96,6] | 51,2 % |
| OSF | F1 maximal | 0,764 [0,679 ; 0,835] | 68,0 % [52,9 ; 81,2] | — |
| OSF | Cible 70 % | 0,754 [0,656 ; 0,836] | 69,7 % [49,2 ; 83,5] | 50,8 % |
| OSF | Cible 80 % | 0,589 [0,478 ; 0,698] | 80,0 % [63,8 ; 89,5] | 50,0 % |
| OSF | Cible 90 % | 0,348 [0,255 ; 0,456] | 90,1 % [80,3 ; 95,5] | 49,6 % |

La fraction sous la cible décrit ces tirages dépendants ; ce n'est pas une
probabilité de risque clinique. Atteindre la cible dans la calibration ne
garantit pas son atteinte dans chaque groupe de nuits laissé de côté.

## Validation croisée du choix du seuil

Repère sur la cible intermédiaire de 80 %, avec une nuit laissée de côté :

| Modèle | Rappel agrégé | Spécificité | Précision | F1 |
|---|---:|---:|---:|---:|
| SleepFM | 79,96 % | 84,18 % | 48,18 % | 60,13 % |
| OSF | 80,00 % | 87,83 % | 54,74 % | 65,00 % |

L'agrégation associe 40 seuils potentiellement différents par règle. Les
percentiles des seuils OSF de cette validation croisée sont **[0,572 ; 0,610]**,
plus resserrés que ceux du bootstrap, où les nuits sont répétées et la
calibration contient moins de nuits distinctes. Aucun seuil de remplacement
n'est retenu. Les huit configurations complètes figurent dans le JSON.

## Vérifications et portée

- Empreintes, partitions, identités et probabilités de validation contrôlées.
  Les huit seuils et matrices sur les 40 nuits complètes sont retrouvés.
- Recalcul indépendant des **500 tirages**, en dupliquant explicitement les
  fenêtres des nuits répétées : seuils et matrices de calibration/OOB identiques.
  Les nuits évaluées sont toujours absentes de leur calibration.
- Les **40 étapes** de validation croisée, leurs seuils et prédictions sont
  reproduits indépendamment ; résumés et percentiles identiques. Archives
  et empreintes du code/plan vérifiées.
- **45 tests ciblés réussis, dont 11 nouveaux** : pondération, ex æquo,
  exclusions de tirages, protection des sources et absence d'utilisation des
  labels de la nuit évaluée pour son propre seuil. Lint, formatage et contrôle
  des espaces du diff réussis. Suite globale et tests lents non relancés.

Les 40 nuits ont déjà servi à plusieurs analyses. Cette étude reste du
développement, avec un modèle figé ; elle ne couvre pas l'incertitude de son
entraînement ni le transfert à d'autres cohortes ou capteurs. Les différences
entre validation et test précédemment observées ne sont pas expliquées
causalement par ce seul diagnostic. Aucun test de supériorité entre modèles
n'est effectué ici.

Révision de base : `2c61add`, avec nouveau code non commité signalé dans
l'environnement du résultat. Code réellement exécuté et plan archivés.

## Preuves conservées

- [Résultats et distributions](../../results/respiratory_threshold_stability_physionet2018.json)
- [Tirages, multiplicités, seuils et matrices](../../results/respiratory_threshold_stability_physionet2018.bootstrap.npz)
- [Prédictions et seuils avec une nuit laissée de côté](../../results/respiratory_threshold_stability_physionet2018.leave_one_out.npz)
- [Archive du code et du plan](../../results/respiratory_threshold_stability_physionet2018.code.zip)

## À faire

- Préciser avec le tuteur et le médecin le coût des erreurs et le niveau de
  garantie attendu avant de retenir un seuil pour un usage.
- Prévoir une calibration plus large et représentative, avec un protocole
  fixé avant son évaluation.
- Confirmer le seuil et le transfert sur une cohorte indépendante dès que
  les accès et usages sont validés. Le constat de l'audit reste ouvert jusque-là.
