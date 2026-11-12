# Des fenêtres aux événements et à l'IAH — plan du 12 novembre 2026

Plan écrit **avant tout calcul**, validé par le tuteur le 12 novembre 2026. Ce n'est pas un
préenregistrement externe. Il prolonge les analyses respiratoires du 12 novembre
(`docs/resultats/respiratoire_*`) sans changer leurs modèles ni leurs seuils.

## Question

À partir des modèles respiratoires figés (logistique sur embeddings SleepFM et OSF), peut-on
**classer chaque nuit selon un IAH de référence ≥ 15 événements/h**, et quelle part des
événements annotés est retrouvée ? C'est le critère de l'étude de faisabilité (cible 1 :
prioriser la relecture des patients suspects).

## Référence

- **IAH de référence** : même définition que l'étude de faisabilité (`Faisabilite_apnee_donnees/compter.py`) :
  apnées obstructives, centrales, mixtes et hypopnées annotées (`.arousal`), comptées
  séparément, divisées par le temps de sommeil expert (N1, N2, N3, REM). Nuits avec au
  moins 1 h de sommeil. La règle d'hypopnée (3 % ou 4 %) n'est pas documentée.
- **Événements de référence** : intervalles fusionnés de `src/data/respiratory_events.py`.

## Prédiction

- **Fenêtres** : toutes les fenêtres de 30 s de chaque nuit, y compris les 0–10 s de
  recouvrement exclues de l'évaluation par fenêtre (à l'usage, aucune annotation ne permet
  de les écarter). Probabilités recalculées avec les modèles figés, sans entraînement ;
  contrôle : identiques aux archives sur les fenêtres déjà évaluées.
- **Seuil** : seuil F1 initial de chaque modèle, déjà fixé sur validation. Aucun nouveau seuil.
- **Indice prédit (principal)** : `x` = fenêtres prédites positives par heure de fenêtres
  évaluées (heure d'enregistrement, pas de sommeil : les stades prédits sur les nuits de
  validation seraient en apprentissage pour les sondes de stades).
- **Calibration** : `IAH estimé = a + b·x`, moindres carrés sur les **40 nuits de
  validation** seulement ; `a`, `b` figés avant le test. Nuit classée positive si
  IAH estimé ≥ 15.
- **Indice secondaire** : moyenne des probabilités par heure, même calibration.

## Évaluation sur les 100 nuits de test, un seul passage

- **Critère principal** : sensibilité et spécificité à IAH ≥ 15, IC 95 % par bootstrap
  des nuits (1 000 tirages, graine 42), pour chaque modèle.
- **Secondaires** : AUC de `x` pour IAH ≥ 15 (indépendante de la calibration) et écart
  apparié OSF − SleepFM ; erreur absolue moyenne et corrélation de Spearman sur l'IAH ;
  biais et limites de Bland-Altman ; κ pondéré sur les 4 classes (< 5, 5–15, 15–30, ≥ 30).
- **Événements (descriptif)** : un événement de référence est retrouvé si une fenêtre
  prédite positive le recouvre d'au moins 10 s ; rappel par type. Séquences prédites sans
  aucun recouvrement : fausses détections par heure. Une séquence peut couvrir plusieurs
  événements : ce n'est pas un appariement un à un.
- Publier le résultat même décevant, avec effectifs, nuits écartées, empreintes, code
  exécuté (commité avant le run) et limites.

**Aucun critère de réussite** : les 100 nuits de test ont déjà servi à six analyses ; le
résultat reste exploratoire et ne permet de retenir aucune règle comme validée.

## Données non utilisées

Ni NSRR, ni les **494 sujets de confirmation** des stades. Aucun téléchargement, aucun GPU.

## Décisions du tuteur (12 novembre 2026)

- Plan validé ; exécution sur PhysioNet 2018 comme décrit ci-dessus.
- Les **494 sujets ne sont pas utilisés** pour la tâche respiratoire. La confirmation se fera
  sur NSRR (MESA ou SHHS) dès l'accès ; la chaîne est préparée d'ici là pour l'appliquer
  rapidement, avec une règle figée avant d'ouvrir ces données.
- Seuil : le seuil F1 initial (« normal »), sans nouvelle sélection.

## Précisions d'exécution (12 novembre 2026, avant tout calcul)

Ajoutées après la validation du tuteur, avant d'écrire le script ; elles fixent des choix
que le plan laissait ouverts, sans en changer les critères.

- **Fenêtres prédites** : toutes celles présentes dans les caches d'embeddings (fenêtres avec
  un stade annoté, identiques pour SleepFM et OSF), labels respiratoires exclus compris. Les
  durées sans fenêtre en cache ne sont ni prédites ni comptées dans les heures.
- **Seuils** : ceux archivés avec les modèles (SleepFM 0,6934 ; OSF 0,7636).
- **Contrôle des probabilités** : sur les fenêtres déjà évaluées, écart maximal ≤ 1e-5 avec
  les archives **et aucune fenêtre classée différemment au seuil** ; sinon arrêt. *Modifié
  avant toute lecture du test :* la tolérance initiale de 1e-6 a arrêté le premier lancement sur
  la validation (tr03-0672, SleepFM, 1,19e-6, arrondi des archives en float32 ; 3 nuits au-dessus
  de 1e-6, aucun franchissement de seuil). Aucun résultat écrit.
- **IAH de référence** : réimplémentation de `compter.py` dans le dépôt ; contrôle : identique
  à `iah_500_sujets_ouverts.csv` (à 0,01 près) pour les 140 nuits utilisées ; sinon arrêt.
- **Nuits écartées** : moins de 1 h de sommeil expert, en validation comme en test.
- **IAH estimé** : `a + b·x` sans troncature ; une valeur négative compte comme < 5.
- **Statistiques** : AUC de Mann-Whitney (ex æquo comptés 0,5) ; IC percentiles 2,5–97,5 ;
  mêmes 1 000 tirages de nuits pour les deux modèles, tirage écarté (et compté) s'il manque une
  classe ; κ à **pondération linéaire** ; Bland-Altman : biais (estimé − référence) ± 1,96 écart-type.
- **Événements** : références individuelles (`individual_events`), retrouvées si l'union des
  fenêtres prédites positives en couvre au moins 10 s. Séquence prédite = fenêtres positives
  d'indices consécutifs ; fausse détection = séquence sans aucun recouvrement avec un événement.

## Limites connues

Fenêtres de 30 s : résolution grossière pour des événements de 10 à 60 s. Indice par heure
d'enregistrement et non de sommeil. PSG de laboratoire, pas polygraphie à domicile. Même
cohorte que le développement. La calibration sur 40 nuits est incertaine ; son incertitude
n'est pas couverte par des IC à paramètres fixes.
