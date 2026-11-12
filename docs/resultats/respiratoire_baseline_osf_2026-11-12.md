# Baseline respiratoire — résultat du 12 novembre 2026

**Première exploration : les embeddings SleepFM contiennent un signal utile
pour reconnaître des fenêtres avec apnées ou hypopnées. La sensibilité reste
limitée et aucun diagnostic individuel n'est validé.**

Plan écrit avant les scores : [RESPIRATORY_BASELINE.md](../plans/Plan%20SleepFM%20respiratoire%2012-11.md).
Résultat complet : [JSON](../../results/respiratory_baseline_physionet2018.json),
avec empreinte du plan et des sources de code réellement exécutées. Le code
était modifié localement ; le run n'est pas présenté comme exécuté sur un commit
propre ni comme une nouvelle confirmation indépendante.

**Suite : OSF obtient une AP de 0,661 contre 0,593 pour SleepFM**, avec un
écart apparié de +0,068 [+0,023 ; +0,116] sur les mêmes fenêtres. La comparaison
reste exploratoire ; détails ci-dessous.

Les plans ont été déplacés dans ce dossier sans modifier leur contenu ni leurs empreintes.

Les [sources exactes exécutées](../../results/respiratory_baseline_physionet2018.code.zip)
sont archivées et correspondent aux empreintes du JSON. Le script courant ajoute
ensuite un garde-fou refusant des sorties dans le dossier des embeddings ; cette
protection des chemins ne change pas le calcul des scores publiés.

## Méthode

- 500 sujets PhysioNet 2018 déjà utilisés pour les stades ; **aucun des 494
  sujets de confirmation** ajouté à cette expérience, aucune donnée NSRR.
- Embeddings SleepFM complets figés, 1 536 dimensions ; fenêtres annotées de
  30 s, alignées sur les temps de l'enregistrement d'origine. Aucun calcul GPU.
- Cible : union des apnées obstructives, centrales, mixtes et des hypopnées
  annotées dans les fichiers WFDB `.arousal`. Les arousals du challenge ne sont
  pas utilisés comme étiquettes d'apnée.
- Positif si au moins 10 s couvertes, négatif si aucune couverture ; entre 0
  et 10 s exclu. Les événements qui se chevauchent ne sont pas comptés deux fois.
- **360 sujets d'ajustement, 40 de validation, 100 de test**, sans chevauchement.
  Logistique pondérée, C = 1, maximum 1 000 itérations, aucun balayage.
- Seuil **0,693377**, choisi par F1 sur la validation seulement. Le modèle
  n'est pas réajusté après ce choix. Convergence atteinte en **113 itérations**.

## Résultat sur le test

**85 230 fenêtres évaluées**, dont 11 438 positives (13,4 %).
**3 902 fenêtres partiellement couvertes exclues**, sur les 89 132 fenêtres
du test de stadification. Les exclusions peuvent faciliter la tâche.

| Mesure | Valeur |
|---|---:|
| **Average precision (AP)** | **0,593 [IC 95 % : 0,521 ; 0,656]** |
| Référence AP à score constant : prévalence | 0,134 |
| AP moins prévalence, IC 95 % | [+0,399 ; +0,514] |
| ROC-AUC | 0,873 |
| F1 binaire | 0,552 |
| Précision | 51,2 % |
| Sensibilité | 59,8 % |
| Spécificité | 91,2 % |

IC par bootstrap de sujets, 500 tirages, graine 42 ; les 500 tirages sont
évaluables. Les mesures sont calculées sur l'ensemble des fenêtres retenues,
pas comme moyenne des scores individuels des nuits.

Matrice de confusion : lignes = référence négative/positive, colonnes =
prédiction négative/positive.

| | Prédit négatif | Prédit positif |
|---|---:|---:|
| Référence négative | 67 272 | 6 520 |
| Référence positive | 4 600 | 6 838 |

**Lecture :** au seuil choisi, environ quatre fenêtres positives sur dix sont
manquées et près de la moitié des alertes sont fausses. Ce sont des mesures
**par fenêtre**, pas une sensibilité de détection d'événements complets ou de
patients. Les scores de la logistique pondérée ne sont pas des risques calibrés.

## Démonstration

[Ouvrir la nuit tr03-0052](../../results/demo_night.html), premier identifiant de
l'archive de test, choisi sans tri des performances. Page HTML autonome :
hypnogrammes expert/logistique/BiGRU, scores de chaque stade, désaccords,
score respiratoire et intervalles annotés. Vue par heure, demi-heure ou nuit
entière ; cliquer pour lire une fenêtre. Aucun signal brut ni serveur requis.

## Vérifications et limites

- Annotations et temps des 500 sujets lus avec succès ; empreintes des sources,
  labels et caches conservées dans [l'inventaire](../../results/respiratory_baseline_physionet2018.labels.json).
- Probabilités et métriques de test sauvegardées ; le modèle `.joblib` et le
  seuil sont disponibles à côté du JSON. Ne charger que cet export de confiance.
- Rechargement sur les 100 nuits : aucune décision différente ; écart maximal
  de probabilité **1,43 × 10⁻⁶** entre calcul global et calcul par nuit (float32).
  AP et ROC-AUC recalculées identiques depuis l'archive.
- Tests rapides, lint et formatage vérifiés. La démonstration a été exécutée
  dans un contrôle DOM JavaScript (trois étendues et sélection d'une fenêtre) ;
  le navigateur local a échoué au lancement, donc le rendu visuel n'est pas attesté ici.
- Les données locales, les modèles de stades, le split de référence et la
  confirmation des 494 sujets ne sont pas modifiés par le parcours.
- Une cohorte, des sujets déjà examinés, une graine et des fenêtres annotées :
  résultat exploratoire. La tâche regroupe plusieurs événements et n'est pas
  spécifique à l'apnée obstructive. Aucun IAH n'est estimé.
- Aucun résultat ne porte sur téléphone, montre ou polygraphe à domicile.
  Les 10 s constituent une règle de labellisation des fenêtres, pas une nouvelle
  règle clinique de cotation des événements.
- Cette expérience ne modifie pas la demande NSRR. L'usage de MESA/SHHS pour
  une tâche différente devra être vérifié dans l'autorisation obtenue.

## Comparaison OSF / SleepFM — même tâche, mêmes fenêtres

[Plan écrit avant les scores OSF](../plans/Plan%20comparaison%20OSF%20SleepFM%20respiratoire%2012-11.md) et
[résultat complet](../../results/respiratory_osf_vs_sleepfm_physionet2018.json).
Les 500 sujets ont été alignés et leurs fichiers vérifiés par empreinte.
OSF est ajusté sur les mêmes 360 nuits, son seuil est choisi sur les mêmes
40 nuits de validation. Les probabilités SleepFM publiées sont conservées,
sans réentraînement. Test : **85 230 fenêtres, 100 nuits**, mêmes exclusions.

| Mesure | SleepFM | OSF |
|---|---:|---:|
| **Average precision (AP)** | **0,593** | **0,661** |
| ROC-AUC | 0,873 | 0,901 |
| F1 | 0,552 | 0,610 |
| Précision | 51,2 % | 59,8 % |
| Sensibilité par fenêtre | 59,8 % | 62,2 % |
| Spécificité par fenêtre | 91,2 % | 93,5 % |

**Critère principal : AP OSF − SleepFM = +0,068073**, IC à 95 %
**[+0,022941 ; +0,115925]**, 1 000 tirages appariés par sujet, graine 42.
Les 1 000 tirages sont évaluables ; la borne basse est au-dessus de zéro.
OSF dépasse donc SleepFM dans cette comparaison exploratoire. Les mesures
avec seuil sont secondaires et ne correspondent pas à une sensibilité imposée
identique : chaque seuil a été choisi par F1 sur la validation.

OSF converge en **164 itérations** ; seuil de validation **0,763604**. Son
IC individuel d'AP, avec les mêmes 1 000 tirages : [0,582473 ; 0,724368].
La sensibilité de 62,2 % laisse encore manquer environ 38 % des fenêtres
positives ; ce résultat ne suffit pas à définir un usage médical.

**Portée :** une graine, une cohorte déjà examinée, quatre canaux OSF absents,
normalisation OSF supposée et dimensions/échelles différentes (768 contre
1 536). Le résultat porte sur les pipelines tels qu'ils sont préparés, sans
isoler l'architecture ou le pré-entraînement. Aucune donnée NSRR supplémentaire
n'est utilisée ; les 494 sujets de confirmation ne sont pas consultés.

[Démonstration comparative, même nuit tr03-0052](../../results/demo_night_comparison.html) :
score respiratoire SleepFM en vert, OSF en violet, seuils de validation et
intervalles annotés. Les scores ne sont pas des probabilités cliniques calibrées.

**Contrôles :** 266 tests rapides réussis, 2 tests XGBoost sautés sur Mac,
7 tests lents désélectionnés ; lint et formatage réussis. Les 32 tests ciblés
couvrent notamment l'alignement, le bootstrap apparié, les entrées modifiées et
la démonstration. Les trois étendues de la page et la sélection d'une fenêtre
sont vérifiées en DOM JavaScript ; le rendu visuel du navigateur reste non attesté.
L'IC principal a été reproduit indépendamment en recopiant explicitement les
nuits tirées avec remise : bornes identiques, écart 0. Le modèle OSF rechargé
ne change aucune décision sur les 100 nuits ; écart maximal de score
5,36 × 10⁻⁷. Les empreintes des artefacts et des sources archivées concordent.

## À faire

Suite descriptive : [analyse des erreurs du 12 novembre](respiratoire_erreurs_2026-11-12.md), par type, stade et nuit, avec les modèles et seuils inchangés.

- Faire choisir la cible précise et l'usage avec le tuteur et le médecin.
- Préparer une validation de la comparaison sur d'autres données, avec les
  capteurs et les droits adaptés à la cible retenue.
- Mesurer ensuite les événements complets et le transfert sur d'autres données,
  sans présenter les sujets déjà consultés comme une nouvelle confirmation.

Source des données et annotations : [PhysioNet Challenge 2018](https://physionet.org/content/challenge-2018/1.0.0/).
