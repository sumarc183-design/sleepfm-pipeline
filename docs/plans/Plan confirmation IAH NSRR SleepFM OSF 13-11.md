# Confirmation de l'IAH ≥ 15 sur NSRR — plan du 13 novembre 2026

**Validé par le tuteur le 13 novembre 2026** (table des canaux, échelle, critère de réussite).
Écrit avant tout accès aux données NSRR
(demande n° 30917 en attente) pour que la confirmation puisse être lancée dès l'ouverture.
Décision du tuteur (12 novembre) : la confirmation se fait sur NSRR, pas sur les 494 sujets.

## Règle figée (aucune modification après l'accès)

Issue du [résultat du 12 novembre](../resultats/respiratoire_iah_2026-11-12.md)
(`results/respiratory_ahi_physionet2018.json`, SHA-256 `eda69eb9…2990`) :

| | SleepFM | OSF |
|---|---|---|
| Modèle respiratoire | `respiratory_baseline_physionet2018.joblib` | `respiratory_osf_vs_sleepfm_physionet2018.joblib` |
| Seuil par fenêtre | 0,6934 | 0,7636 |
| Calibration `IAH = a + b·x` | a = 1,992 ; b = 0,869 | a = −1,898 ; b = 1,025 |
| Nuit signalée si | IAH estimé ≥ 15 | IAH estimé ≥ 15 |

`x` = fenêtres de 30 s prédites positives par heure de fenêtres disponibles. Aucun
réentraînement, aucun nouveau seuil, aucune recalibration sur NSRR.

## Données

- **Cohorte** : MESA d'abord (cohorte indépendante, adultes), SHHS ensuite si accordée.
  Une seule visite par participant (SHHS-1 seulement).
- **Sélection** : toutes les nuits avec annotations XML NSRR et au moins 1 h de sommeil.
  Liste figée et empreintée **avant** toute prédiction ; nuits écartées publiées avec leur motif.
- **Annotations** : `src/data/nsrr_annotations.py` (stades R&K regroupés en AASM, 4 types
  d'événements). Un libellé inconnu arrête la lecture : à vérifier sur les premiers fichiers.

## Référence

- **Principale** : même règle que PhysioNet 2018 et l'étude de faisabilité, tous les
  événements annotés des 4 types par heure de sommeil expert.
- **Secondaire** : IAH officiel NSRR à 3 % et 4 % de désaturation (variables du dictionnaire,
  par exemple `ahi_a0h3` et `ahi_a0h4`, **noms à vérifier**). La règle d'hypopnée de PhysioNet
  2018 n'étant pas documentée, l'écart entre les deux références sera rapporté.

## Évaluation (un seul passage)

- **Critère principal** : sensibilité et spécificité à IAH ≥ 15 pour chaque modèle, IC 95 %
  par bootstrap des participants (1 000 tirages, graine 42).
- **Secondaires** : AUC de `x` et écart apparié OSF − SleepFM ; erreur absolue moyenne,
  Spearman, Bland-Altman, κ pondéré linéaire sur 4 classes ; mêmes calculs avec la référence
  officielle NSRR.
- **Critère de réussite** (validé le 13 novembre) : pour chaque modèle séparément, la borne
  basse de l'IC 95 % de la sensibilité et celle de la spécificité au-dessus de 60 %, avec la
  référence principale.

## Préparation (validée le 13 novembre)

1. **Canaux** (`src/data/nsrr_channels.py`, libellés EDF à vérifier sur les premiers fichiers).
   Règle validée : reproduire l'entrée PhysioNet. Un canal nul sur PhysioNet reste nul même s'il
   existe ; un canal absent de NSRR est mis à zéro et signalé.

   | Entrée | PhysioNet 2018 | MESA | SHHS-1 |
   |---|---|---|---|
   | EEG C3-M2 | oui | **zéros** | EEG(sec) |
   | EEG C4-M1 | oui | EEG3 | EEG |
   | EEG F3, F4, O1, O2 (SleepFM) | oui | **zéros** | **zéros** |
   | EOG gauche | E1-M2 | EOG-L | EOG(L) |
   | EMG menton | oui | EMG | EMG |
   | Sangles thorax / abdomen | oui | Thor / Abdo | THOR RES / ABDO RES |
   | Débit (`AIRFLOW`, `NP` d'OSF) | capteur non documenté | Flow (thermistance) | AIRFLOW (thermistance) |
   | SaO2 (SleepFM) | oui | SpO2 | SaO2 |
   | ECG | oui | EKG | ECG |
   | EOG droit, jambes, ronflement (OSF) | zéros | zéros (gardés nuls) | zéros |

   MESA perd 5 des 7 voies EEG de SleepFM et l'EEG C3 d'OSF ; SHHS-1 en perd 4 pour SleepFM
   et aucune pour OSF. On peut donc s'attendre à une perte plus forte sur MESA, surtout pour SleepFM.
2. **Échelle** : SleepFM reçoit les valeurs brutes du .mat PhysioNet, sans normalisation.
   D'après les en-têtes, elles valent 1 unité par µV pour l'EEG, l'EOG, l'EMG et l'ECG, et
   655,35 × % − 32768 pour la SaO2. Les EDF NSRR seront remis à cette échelle. Les sangles et
   le débit n'ont pas d'unité comparable d'un appareil à l'autre. **Règle validée par le tuteur
   (13 novembre)** : l'échelle robuste de chaque nuit est ramenée à la médiane des 360 nuits
   d'apprentissage PhysioNet, 173 (ABD), 83 (CHEST), 13 (AIRFLOW)
   ([mesure du 13 novembre](../resultats/respiratoire_amplitude_sangles_2026-11-13.md) ; SleepFM
   a déjà vu des écarts d'échelle d'un facteur 12 à 800 entre nuits). Code :
   `physionet_median_scale` (`--respiratory-rule physionet_median_v1`) ; un canal d'échelle nulle
   est refusé. OSF n'est pas concerné, car il normalise déjà chaque canal sur la nuit.
3. **Fréquences** : rééchantillonnage vers 256 Hz (SleepFM) et 64 Hz (OSF) avec le même code
   que PhysioNet. La SpO2 NSRR est à 1 Hz et les sangles SHHS à 10 Hz, alors que PhysioNet
   enregistre à 200 Hz : le contenu fréquentiel diffère, ce qui reste une limite.
4. **Calcul** : extraction des embeddings sur Kaggle (GPU) ; volume MESA à estimer.
5. **Différences de population** : MESA à domicile, adultes plus âgés ; la calibration peut
   se déplacer. C'est précisément ce que la confirmation mesure : aucun ajustement.

## Limites connues

Règle développée sur 140 nuits d'une seule cohorte ; IAH estimé imprécis (limites de
Bland-Altman au-delà de ±20 /h sur PhysioNet). Changement de capteurs et de population.

## Exécution (précisions du 13 novembre, sans changement de décision)

0. `scripts/freeze_nsrr_subjects.py` : liste des nuits ayant un EDF et un XML NSRR, à partir des
   noms de fichiers seulement (dossier ou index du site NSRR), commitée avant toute extraction.
   Les fichiers sans partenaire sont publiés dans la liste.
1. `scripts/embed_nsrr.py` : embeddings SleepFM et OSF des nuits de la liste figée
   (`--respiratory-rule physionet_median_v1`), sur Kaggle (`notebooks/sleepfm_nsrr_embeddings.ipynb` :
   EDF téléchargés par lots puis supprimés, faute de place pour toute la cohorte).
2. `scripts/evaluate_nsrr_ahi.py` : un seul passage. Le script vérifie l'empreinte de la règle
   figée, les modèles, les seuils et les poids des encodeurs (ceux de PhysioNet), et refuse une
   extraction incomplète. Les nuits refusées à l'extraction, à la lecture des annotations ou
   avec moins de 1 h de sommeil sont publiées avec leur motif. Il écrit ensuite les résultats,
   le verdict du critère pour chaque modèle et une archive du code. Une sortie existante n'est
   jamais écrasée.

Vérifié le 13 novembre sur le site public NSRR, sans accès aux données :
- Liste figée depuis l'index public des fichiers (noms seulement) : MESA 2 056 nuits
  (`configs/nsrr_mesa_subjects.json`, SHA-256 `aef9d758…1fe7`, 402 Go d'EDF), SHHS-1 5 793 nuits
  (`configs/nsrr_shhs1_subjects.json`, SHA-256 `c06453bd…4baf`, 228 Go). Chaque EDF a son XML.
- Chemins : `mesa/polysomnography/edfs/`, `…/annotations-events-nsrr/` ; SHHS-1 dans les
  sous-dossiers `shhs1/`.
- Référence secondaire : `ahi_a0h3` et `ahi_a0h4` existent dans les deux cohortes (hypopnées avec
  baisse du débit > 30 % et désaturation ≥ 3 % ou ≥ 4 %). CSV : `mesa-sleep-dataset-0.8.0.csv`,
  `shhs1-dataset-0.21.0.csv` ; l'identifiant numérique est relié au nom de fichier
  (`mesa-sleep-0001` → 1, `shhs1-200001` → 200001).
- Restent à vérifier sur les premiers fichiers : libellés EDF, libellés d'événements XML.

Contrôles ajoutés après l'audit du 13 novembre (sans changement de règle, de modèle, de seuil ni de critère) :
- **Précontrôle sans cible ni prédiction** : code et plan commités, règle figée, liste approuvée
  (empreinte ci-dessus) et couverture exacte (chaque nuit extraite ou rejetée avec motif, aucune
  nuit étrangère ni en double), contrat d'extraction complet (version, paramètres d'échelle, code de
  préparation), chaque cache relu et empreinté, concepts XML connus, CSV officiel valide.
- **Passage unique** : marqueur écrit dans `results/nsrr_runs/` (cohorte, liste, règle) avant la
  première lecture des cibles ; un autre préfixe ou une relance après échec est refusé.
- **Exclusions fixées avant le passage** : XML illisible ; annotations au-delà de la fin du signal
  de plus d'une époque (30 s) ; moins de 1 h de sommeil. La dernière époque incomplète de l'EDF est
  écartée à la préparation. Un concept respiratoire inconnu arrête le précontrôle (correspondance
  à décider avant le passage) ; désaturations et artefacts d'oxymétrie sont explicitement non ciblés.
- **Référence officielle** : `ahi_a0h3` et `ahi_a0h4` lues ensemble (identifiant `mesaid` pour MESA,
  `nsrrid` pour SHHS), mêmes métriques que la référence principale avec IC bootstrap ; valeurs
  manquantes comptées, valeurs négatives ou identifiants non entiers refusés.
- **Question ouverte au tuteur** : conduite si très peu de nuits positives ou négatives sont
  évaluables (IC bootstrap dégénérés), à fixer avant le passage.
