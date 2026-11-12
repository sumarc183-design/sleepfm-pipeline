# SleepFM / OSF — séquences respiratoires, 12 novembre 2026

**Les fausses fenêtres positives se regroupent partiellement, mais restent
souvent isolées.** Pour OSF au seuil initial, 4 777 fenêtres donnent **3 142
séquences**, dont **73,1 % d'une seule fenêtre**. Au seuil visant 80 % de
sensibilité sur validation, 8 817 fenêtres donnent **5 138 séquences**.
Ce sont des séquences de fenêtres, pas des alertes cliniques évaluées.

## Périmètre et définition

Même test déjà exploré : 100 nuits, 85 230 fenêtres évaluées, soit **710,25
heures nominales**. Les fenêtres exclues restent exclues. Modèles, probabilités
et huit seuils précédents conservés ; aucun entraînement ni téléchargement,
aucune utilisation de NSRR ou des 494 sujets de confirmation.

Une séquence exige la même nuit, des indices d'époque consécutifs et des
débuts espacés de 30 s, à 0,01 s près. Fenêtre non sélectionnée, exclusion ou
lacune : la séquence s'arrête. Aucun comblement ni filtre de durée.

## Charge des séquences de faux positifs

Le taux utilise les seules heures de fenêtres évaluées : ce n'est pas un taux
par heure de sommeil ou par heure d'enregistrement complet. Les cibles 70/80/90 %
sont celles de la validation, pas les sensibilités obtenues sur le test.

| Modèle | Seuil | Fausses fenêtres positives | Séquences de faux positifs | Séquences / heure évaluée [IC 95 %] | Une seule fenêtre |
|---|---|---:|---:|---|---:|
| SleepFM | F1 initial | 6 520 | 4 382 | 6,17 [5,18 ; 7,21] | 74,9 % |
| SleepFM | Cible 70 % | 8 316 | 5 308 | 7,47 [6,47 ; 8,49] | 72,7 % |
| SleepFM | Cible 80 % | 14 300 | 7 902 | 11,13 [10,13 ; 12,17] | 65,3 % |
| SleepFM | Cible 90 % | 25 278 | 10 779 | 15,18 [14,20 ; 16,12] | 56,0 % |
| OSF | F1 initial | 4 777 | 3 142 | 4,42 [3,62 ; 5,27] | 73,1 % |
| OSF | Cible 70 % | 4 955 | 3 245 | 4,57 [3,75 ; 5,43] | 72,8 % |
| OSF | Cible 80 % | 8 817 | 5 138 | 7,23 [6,26 ; 8,18] | 67,8 % |
| OSF | Cible 90 % | 16 648 | 7 819 | 11,01 [10,08 ; 12,10] | 58,4 % |

IC par bootstrap des 100 nuits, 500 tirages, graine 42, seuils fixes. Ils ne
couvrent pas l'incertitude du choix des seuils. Les sensibilités diffèrent
entre configurations : ces taux ne sont pas une comparaison à rappel identique.

Pour OSF, la médiane du nombre de séquences de faux positifs par nuit passe
de **26 à 49** entre le seuil initial et la cible de validation 80 %. Leur
durée médiane reste **30 s**, leur 90e percentile **90 s** ; le maximum passe
de **10,5 à 13 minutes**. Durées nominales à 30 s par fenêtre ; bornes observées
également conservées. Ce regroupement ne supprime aucune décision du modèle.

## Deux décomptes à distinguer

Au seuil OSF initial, les fenêtres prédites positives forment **5 566 séquences** :
2 463 entièrement négatives dans les annotations, 2 510 entièrement positives
et 593 mixtes. Les **3 142 séquences de faux positifs** comptées plus haut
incluent aussi les portions négatives des séquences mixtes.

Ni une séquence mixte ni une séquence entièrement positive ne constitue un
événement respiratoire apparié et validé. Aucun rappel événementiel, IAH,
diagnostic ou charge d'alertes cliniques n'a été calculé.

## Contrôles et ordre temporel

Le [plan initial](../plans/Plan%20s%C3%A9quences%20respiratoires%20SleepFM%20OSF%2012-11.md)
a été arrêté avant tout décompte : les archives suivaient les noms des fichiers
(`1, 10, 100…`), pas le temps. Le [plan v2](../plans/Plan%20s%C3%A9quences%20respiratoires%20SleepFM%20OSF%2012-11%20v2.md)
explicite donc le tri conjoint par nuit et indice entier, avant l'analyse.
Les définitions de séquences sont conservées. Les 100 nuits nécessitaient
ce tri ; la permutation est archivée et aucune source n'a été modifiée.

- Empreintes des références et prédictions vérifiées ; les huit matrices,
  nombres de faux positifs et de fenêtres positives prédites sont retrouvés.
- Reconstruction indépendante des **111 316 séquences** des huit configurations :
  bornes, classes, durées et couverture exactes, sans lacune franchie. Permutation
  vérifiée contre les lignes sources. IC recalculés par tirage explicite des
  nuits : écart maximal **5,4 × 10⁻¹⁵**.
- **30 tests ciblés réussis, dont 13 nouveaux** : lacunes, exclusions, ordre,
  séquences mixtes, sources modifiées et absence de prédiction. Lint, formatage
  et contrôle des espaces du diff réussis. Suite globale et tests lents non relancés.

Révision de base : `921758d223b28579bcb25d33c50e72c5a589fc2c`, avec modifications
locales non commitées. Le code réellement exécuté et le plan v2 sont archivés.

## Preuves conservées

- [Résultats complets et comptes par nuit](../../results/respiratory_runs_physionet2018.json)
- [Intervalles de toutes les configurations](../../results/respiratory_runs_physionet2018.intervals.npz)
- [Permutation vers les lignes d'origine](../../results/respiratory_runs_physionet2018.ordering.npz)
- [Code et plan](../../results/respiratory_runs_physionet2018.code.zip)
- [Arrêt du premier contrôle](../../results/respiratory_runs_precheck.json)
- [Compromis sensibilité/précision précédent](respiratoire_seuils_2026-11-12.md)

## À faire

- Définir avec le tuteur et le médecin l'usage et ce qui doit constituer une
  alerte avant de choisir une règle de regroupement ou de temporisation.
- Si un filtre de durée est envisagé, mesurer sur validation les événements
  manqués en plus de la réduction des séquences ; ne pas le choisir sur ce test.
- Apparier ensuite les sorties aux événements complets et confirmer sur des
  données indépendantes. Le simple regroupement de fenêtres ne suffit pas.
