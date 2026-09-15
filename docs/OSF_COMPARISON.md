# Comparer SleepFM et OSF — ce qui n'est PAS un simple remplacement de modèle

`src/models/osf_vit1d.py` charge le vrai backbone OSF (vendé depuis
yang-ai-lab/OSF-Open-Sleep-FM, MIT). Testé : architecture (ViT-Base 1D,
~86M paramètres), forward pass, round-trip de checkpoint. Mais comparer
"SleepFM vs OSF" n'est pas neutre techniquement :

## 1. Conventions de canaux différentes

| | SleepFM | OSF |
|---|---|---|
| Nb canaux | 13 | 12 |
| Fréquence | 256 Hz | 64 Hz |
| Épine | montage PSG nommé (F3-M2, C3-M2, ABD, SaO2, ECG...) | ECG, EMG_Chin/LLeg/RLeg, ABD, THX, NP, SN, EOG_E1/E2, EEG_C3/C4 |

Aucune correspondance 1:1 automatique. Faire tourner OSF sur nos données
PhysioNet 2018 nécessite d'écrire un mapping canal-par-canal (ex. ABD/ECG
se retrouvent des deux côtés, mais SaO2 et AIRFLOW n'ont pas d'équivalent
direct côté OSF ; inversement THX/NP/SN n'existent pas nommément côté
SleepFM/PhysioNet 2018) — **pas encore fait**, à traiter avant tout
chiffre comparatif.

## 2. Schémas de labels différents pour la stadification

- SleepFM : 5 classes (Wake, N1, N2, N3, REM)
- OSF (benchmark publié) : 4 classes (Awake, Light Sleep, Deep Sleep, REM)

"Light Sleep" chez OSF regroupe vraisemblablement N1+N2. Pour comparer les
deux modèles équitablement, il faut évaluer les deux sur le **même**
schéma (typiquement : fusionner N1+N2 côté SleepFM avant de calculer le
kappa), plutôt que de comparer des chiffres publiés sur des découpages de
classes différents.

## 3. Tâches communes déjà alignées (bon signe)

Le benchmark OSF inclut Arousal / Hypopnea / Oxygen Desaturation — ce sont
exactement les tâches de différenciation phase 2 envisagées dans
`docs/ROADMAP.md`. Une fois la stadification reproduite, ces tâches sont
un point de comparaison direct sans problème de schéma de labels (détection
binaire dans les deux cas).

## À faire avant de publier un chiffre "SleepFM vs OSF"

1. Écrire le mapping de canaux PhysioNet 2018 -> format d'entrée OSF (12ch, 64Hz)
2. Choisir le schéma de labels commun (4 classes, en fusionnant N1+N2 côté SleepFM)
3. Évaluer les deux sur exactement les mêmes epochs de test
