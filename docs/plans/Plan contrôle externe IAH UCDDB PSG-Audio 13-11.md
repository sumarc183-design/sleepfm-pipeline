# Plan — contrôle externe de l'IAH sur UCDDB et PSG-Audio (13 novembre 2026)

**Statut : validé par le tuteur le 13 novembre 2026, qui laisse les choix ouverts à Marc ; décisions ci-dessous. Aucun calcul lancé.**

## Objectif

Appliquer la règle IAH figée sur PhysioNet 2018 (`results/respiratory_ahi_physionet2018.json`,
SHA-256 `eda69eb9…2990`) à deux bases ouvertes, sans attendre NSRR. Il s'agit d'un **contrôle
complémentaire** : la confirmation reste celle sur NSRR (décision du 12 novembre). Rien n'est
réajusté : mêmes modèles, seuils, calibration, règle d'échelle et contrôles que pour NSRR.

## Données (vérifiées sur les pages publiques, sans ouvrir de nuit)

| | UCDDB (PhysioNet) | PSG-Audio (Sismanoglio, Athènes) |
|---|---|---|
| Nuits | 25 adultes adressés pour suspicion d'apnée (21 hommes, 4 femmes) | 212 patients (Scientific Data 2021) ; une version à 284 patients est citée, à vérifier |
| Licence | Open Data Commons Attribution 1.0 | Article sous CC BY 4.0 ; conditions de ScienceDB à confirmer sur la page du dépôt |
| Notation | Un technicien ; stades R&K ; apnées obstructives, centrales, mixtes, hypopnées (+ respiration périodique) | Technicien puis médecin ; règles AASM ; mêmes 4 types d'événements |
| Fichiers | EDF `.rec` (en-tête lu : 6,2 h pour la 1re nuit), annotations texte `_stage` et `_respevt` | EDF découpés en tranches de 1 h, annotations `.rml` (stades et événements) |
| Particularité | Définition des hypopnées non documentée | Protocole en deux temps pour certains : environ 4 h de diagnostic, puis réglage de la PPC |

## Correspondance des canaux (même logique que NSRR)

| Entrée SleepFM | UCDDB | PSG-Audio |
|---|---|---|
| C3-M2 / C4-M1 | `C3A2` / `C4A1` (128 Hz) | `EEG C3-A2` / `EEG C4-A1` (200 Hz) |
| E1-M2 | `Lefteye` (64 Hz) | `EOG LOC-A2` |
| Chin1-Chin2 | `EMG` (64 Hz) | `EMG Chin` |
| ABD / CHEST | `abdo` / `ribcage` (8 Hz) | `Effort ABD` / `Effort THO` (100 Hz) |
| AIRFLOW | `Flow`, thermistance (8 Hz) | `Flow Patient`, thermistance (100 Hz) |
| SaO2 | `SpO2` (8 Hz) | `SpO2` (1 Hz) |
| ECG | `ECG` (128 Hz) | `ECG I` |
| F3, F4, O1, O2 | absents : zéros | absents : zéros |

Sangles et débit : règle `physionet_median_v1`, déjà validée. OSF reçoit les mêmes canaux que
pour NSRR ; les canaux nuls sur PhysioNet restent nuls.

## Référence, exclusions, passage

- **Référence principale** : 4 types d'événements par heure de sommeil expert, comme pour PhysioNet
  et NSRR. La respiration périodique (UCDDB) n'est pas comptée.
- **Exclusions** fixées avant calcul : annotations illisibles, moins de 1 h de sommeil,
  annotations au-delà de la fin du signal. Un libellé inconnu arrête le précontrôle.
- **Un seul passage par base**, liste des nuits figée avant, avec le même précontrôle et le même
  marqueur de passage que pour NSRR.

## Points à vérifier sur les premiers fichiers (avant tout résultat)

1. Unités des EEG : « NV » dans l'en-tête UCDDB, « a.u. » dans PSG-Audio. SleepFM attend des µV :
   l'échelle doit être établie avant le passage, sinon l'EEG est faux pour SleepFM.
2. PSG-Audio : deux canaux portent le même libellé `Flow Patient` (thermistance et canule) ; ils
   seront distingués par leur position, à vérifier.
3. PSG-Audio : repérer le début du réglage de la PPC dans les `.rml`, pour ne garder que la partie
   diagnostique.

## Vérifications du 13 novembre (en-têtes, libellés et pages publiques ; aucun signal ni compte lu)

- **UCDDB, échelle** : EEG, EOG, EMG et ECG **non calibrés**. L'en-tête donne 0 à 1 pour toute la
  plage numérique (12 bits), avec l'unité « NV ». L'amplitude réelle est inconnue.
- **UCDDB, annotations** : événements horodatés en heure d'horloge, à recaler sur le début de l'EDF.
  Types `APNEA-O`, `APNEA-C`, `APNEA-M`, `HYP-O`, `HYP-C`, `HYP-M` ; les trois `HYP-*` comptent
  comme hypopnées et la colonne respiration périodique n'est pas comptée. Stades (ancienne page
  PhysioNet) : 0 éveil, 1 REM, 2 à 5 stades 1 à 4, 6 artefact et 7 indéterminé (non notés).
- **PSG-Audio, dépôt** (page ScienceDB lue dans le navigateur) : CC BY 4.0 confirmée, version 3 du
  14 mars 2022, 212 patients. **918 Go**, car le son à 48 kHz est inclus dans les EDF.
  Annotations des médecins dans `APNEA_RML` ; `APNEA_RML_clean` retire automatiquement des apnées
  jugées fausses. Le téléchargement demande peut-être un compte ScienceDB.

## Décisions (13 novembre, accord du tuteur)

1. Les deux bases sont ajoutées en contrôle complémentaire ; la confirmation reste NSRR.
2. **UCDDB : descriptif seulement** (25 nuits, IC trop larges pour un critère). **PSG-Audio : même
   critère que NSRR**, pour chaque modèle : bornes basses des IC 95 % de la sensibilité et de la
   spécificité au-dessus de 60 %, référence principale.
3. **PSG-Audio : partie diagnostique seulement**, avant le début du réglage de la PPC ; l'IAH porte
   sur cette partie. Les 3 points à vérifier ci-dessus sont réglés sur les premiers fichiers, avant
   de figer les listes ; une règle qui ne peut pas être établie sans regarder les résultats exclut
   la base plutôt que d'être devinée.
4. **UCDDB, échelle de SleepFM** : la règle validée pour les sangles (échelle robuste ramenée à la
   médiane PhysioNet) est étendue aux voies non calibrées (C3, C4, EOG gauche, EMG, ECG). Il faut
   d'abord les médianes PhysioNet de ces voies, calculées en descriptif sur les 360 nuits
   d'apprentissage, sans label ni modèle, comme le 13 novembre. OSF n'est pas concerné.
5. **PSG-Audio, référence** : annotations des médecins (`APNEA_RML`), pas la version nettoyée
   automatiquement, qui serait une autre référence.
6. **PSG-Audio, ordre** : UCDDB d'abord (petit et immédiat). PSG-Audio ensuite sur Kaggle, patient
   par patient (EDF supprimés après lecture, comme NSRR), après vérification de l'accès.
7. **UCDDB, libellés trouvés dans les 25 fichiers** (libellés seuls, aucun compte ni résultat) :
   `POSSIBLE` (événement incertain) et `PB EVENT` (respiration périodique) ne sont pas comptés ;
   le code de stade 8, non documenté (2 nuits), est traité comme non noté. Les 25 EDF passent la
   correspondance des canaux ; l'en-tête de certaines nuits diffère seulement sur des voies non
   utilisées (« Soud », jambes).

## Exécution

- Liste figée : `configs/ucddb_subjects.json` (25 nuits, SHA-256 `ee768b21…5411`), index public
  PhysioNet.
- Code : `scripts/embed_nsrr.py --cohort ucddb` puis `scripts/evaluate_nsrr_ahi.py --cohort ucddb
  --plan <ce fichier>` (précontrôle, passage unique, pas de verdict).
- SleepFM attend les médianes PhysioNet des voies électriques ([plan d'amplitude](Plan%20amplitude%20EEG%20EOG%20EMG%20ECG%20PhysioNet%2013-11.md)).

## Correction du 16 novembre (accord du tuteur)

Le premier passage UCDDB (règle v1) a révélé un défaut de préparation : les voies mises à
l'échelle n'étaient pas recentrées, alors que celles de PhysioNet sont centrées sur 0 ([compte
rendu](Compte%20rendu%20contr%C3%B4le%20externe%20UCDDB%2013-11.md)). Règle **v2** : recentrage
sur la médiane de la nuit, puis mise à l'échelle. Elle s'applique à NSRR, PSG-Audio et à un
**second passage UCDDB déclaré, analyse exploratoire sur une cohorte déjà ouverte** (audit du 15 novembre), refait en entier (nouvelle extraction, marqueur distinct). Les
deux résultats sont publiés et rien d'autre ne change. Le défaut a été trouvé en voyant un
résultat anormal : le second passage reste donc descriptif.

## PSG-Audio — vérifications et exécution (16 novembre)

Vérifié sur le dépôt (index public, en-tête d'une EDF, deux `.rml`, README ; aucun résultat) :
- **287 patients** dans la version 3 (contre 212 dans l'article), liste figée :
  `configs/psgaudio_subjects.json` (SHA-256 `c81c004c…4e10`).
- **PPC** : le patient 1006 (protocole avec PPC) n'a qu'une session de 4 h, soit la partie
  diagnostique seule. Rien à découper ; la durée de session sert de fin d'enregistrement.
- **EEG, EOG, EMG, ECG en µV** (calibrés, conversion directe) ; sangles et débit en règle v2.
- **Débit** : deux voies `Flow Patient`, la thermistance est choisie par son champ capteur.
- **Annotations** : stades des médecins (`UserStaging`, AASM), jamais la notation automatique ;
  Cheyne-Stokes, respiration périodique et RERA non comptés.
- **Téléchargement** : le serveur sert les plages multiples ; seules les voies non audio sont
  récupérées (environ 100 Mo par patient au lieu de 3 Go), tranches recollées après contrôle de
  continuité, MD5 du `.rml` vérifié (`scripts/download_psg_audio.py`).
