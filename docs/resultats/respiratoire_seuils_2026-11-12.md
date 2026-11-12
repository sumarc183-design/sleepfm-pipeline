# SleepFM / OSF — compromis des seuils respiratoires, 12 novembre 2026

**Abaisser le seuil retrouve davantage de fenêtres positives et d'hypopnées,
mais augmente les faux positifs.** Pour OSF, le seuil visant 80 % de sensibilité
sur la validation atteint **74,1 % sur le test**, contre 62,2 % au seuil initial.
Les fausses fenêtres positives passent de **4 777 à 8 817** ; le rappel des
hypopnées de **48,9 % à 63,3 %**. Ce compromis reste à discuter selon l'usage.

## Méthode

Deux modèles respiratoires existants, sans réentraînement. Les seuils sont
choisis sur les **40 nuits de validation, 34 437 fenêtres**, puis appliqués
aux probabilités archivées des **100 nuits de test, 85 230 fenêtres**.
Les 3 902 fenêtres exclues du test initial restent exclues.

Trois cibles fixées dans le [plan](../plans/Plan%20seuils%20respiratoires%20SleepFM%20OSF%2012-11.md) :
70 %, 80 % et 90 % de sensibilité sur validation. Le plus grand seuil qui
atteint la cible est retenu, ex æquo inclus. Les six seuils sont arrêtés avant
leurs métriques de test. Les seuils F1 initiaux restent inchangés.

## Résultats sur le test

Les cibles de validation ne garantissent pas la même sensibilité sur le test.
« Hypopnées » désigne le groupe exclusif de fenêtres positives d'hypopnée seule,
déjà défini lors de l'analyse des erreurs (6 238 fenêtres).

| Modèle | Règle sur validation | Sensibilité test | Spécificité | Précision | Rappel hypopnées | Fausses fenêtres positives |
|---|---|---:|---:|---:|---:|---:|
| SleepFM | F1 initial | 59,8 % | 91,2 % | 51,2 % | 54,1 % | 6 520 |
| SleepFM | Cible 70 % | 65,1 % | 88,7 % | 47,2 % | 60,5 % | 8 316 |
| SleepFM | Cible 80 % | 76,8 % | 80,6 % | 38,1 % | 74,2 % | 14 300 |
| SleepFM | Cible 90 % | 88,2 % | 65,7 % | 28,5 % | 87,4 % | 25 278 |
| OSF | F1 initial | 62,2 % | 93,5 % | 59,8 % | 48,9 % | 4 777 |
| OSF | Cible 70 % | 63,1 % | 93,3 % | 59,3 % | 49,9 % | 4 955 |
| OSF | Cible 80 % | 74,1 % | 88,1 % | 49,0 % | 63,3 % | 8 817 |
| OSF | Cible 90 % | 85,3 % | 77,4 % | 36,9 % | 78,0 % | 16 648 |

À cible de validation 80 %, les sensibilités de test diffèrent entre modèles :
on ne compare donc pas leurs faux positifs à sensibilité de test identique.
Aucun seuil n'a été réajusté pour modifier cette comparaison.

Repère d'incertitude pour cette cible intermédiaire :

| Modèle | Sensibilité test, IC 95 % | Précision, IC 95 % | Rappel hypopnées, IC 95 % |
|---|---|---|---|
| SleepFM | 76,8 % [72,4 ; 80,6] | 38,1 % [33,2 ; 43,1] | 74,2 % [69,8 ; 78,4] |
| OSF | 74,1 % [68,0 ; 80,3] | 49,0 % [43,4 ; 54,1] | 63,3 % [55,5 ; 71,0] |

Bootstrap par nuit, 500 tirages, graine 42. Ces IC gardent les seuils fixes :
ils ne couvrent pas l'incertitude de leur sélection sur les 40 nuits de validation.
Le JSON contient tous les seuils, F1, matrices et intervalles.

## Vérifications et limites

- Empreintes des modèles, références, labels et caches des 40 nuits de validation
  vérifiées ; fenêtres de test identiques aux archives précédentes.
- Les deux seuils F1 recalculés sur validation retrouvent **exactement** les
  valeurs initiales. Les scores de test et rappels hypopnées initiaux sont retrouvés.
- Recalcul indépendant des six seuils avec les courbes précision/rappel ;
  toutes les métriques contrôlées. IC des huit configurations reproduits en
  recopiant explicitement les nuits tirées avec remise : **écart nul**.
- **53 tests ciblés réussis, dont 17 nouveaux** ; lint, formatage et contrôle
  des espaces du diff réussis. La suite globale et les tests lents n'ont pas
  été relancés pour ce complément.

Ce sont des fenêtres sur PSG annotée, pas des événements, un IAH ou un diagnostic.
La précision dépend de la prévalence ; les scores ne sont pas des risques
cliniques calibrés. Les nuits de test sont déjà explorées et les limites OSF
restent les mêmes (quatre canaux absents, normalisation supposée). Aucune
nouvelle donnée, NSRR ou consultation des 494 sujets de confirmation.

Révision de base : `921758d223b28579bcb25d33c50e72c5a589fc2c` ; nouveau script
non commité, identifié par ses empreintes et son archive. Le champ d'environnement
`code_modified` ne compte que les changements aux fichiers déjà suivis par Git.

## Preuves conservées

- [Résultats complets et seuils](../../results/respiratory_thresholds_physionet2018.json)
- [Probabilités de validation](../../results/respiratory_thresholds_physionet2018.validation.npz)
- [Inventaire des entrées de validation](../../results/respiratory_thresholds_physionet2018.inputs.json)
- [Code exécuté et plan](../../results/respiratory_thresholds_physionet2018.code.zip)

## À faire

Complément réalisé le même jour : [regroupement temporel des fenêtres respiratoires](respiratoire_sequences_2026-11-12.md), sans nouvelle règle d'alerte ni modification des seuils.

- Faire préciser avec le tuteur et le médecin le coût acceptable des événements
  manqués et des fausses alertes avant de retenir un seuil pour un usage.
- Garder la référence actuelle ; choisir toute nouvelle règle sur validation,
  puis la confirmer sur une cohorte indépendante.
- Évaluer ensuite les événements complets et la charge d'alertes réelle :
  une succession de fenêtres faussement positives ne se compte pas directement
  comme autant d'alertes indépendantes.
