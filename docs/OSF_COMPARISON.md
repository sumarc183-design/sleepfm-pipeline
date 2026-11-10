# Comparer SleepFM et OSF — ce qui n'est PAS un simple remplacement de modèle

`src/models/osf_vit1d.py` implémente le backbone OSF adapté de
yang-ai-lab/OSF-Open-Sleep-FM (MIT). Testé : architecture (ViT-Base,
~86M paramètres), forward pass, round-trip de checkpoints synthétiques. Mais comparer
"SleepFM vs OSF" n'est pas neutre techniquement :

## Compatibilité du code vérifiée le 9 novembre 2026

Les noms de poids de la projection d'attention suivent le code officiel
(`to_out.0.weight` / `bias`). Les deux modes `lead_wise=0` et `lead_wise=1`
sont pris en charge ; le second regroupe les canaux par `patch_size_ch`, conserve
`lead_emb` pour charger les poids, et ne l'ajoute pas aux tokens. La forme et
les valeurs finies des entrées sont contrôlées ; les métadonnées incohérentes
ou les poids manquants restent refusés par défaut.

Une comparaison indépendante avec la
[source officielle figée au commit d75d616](https://github.com/yang-ai-lab/OSF-Open-Sleep-FM/blob/d75d616736a52733d8565edba9e0fd07dab41c4a/osf/backbone/vit1d_cls.py)
charge strictement les mêmes poids aléatoires dans les deux implémentations :
pour les deux modes, les tokens sont identiques sur les tenseurs synthétiques
testés (erreur maximale 0). Ce contrôle utilise un modèle réduit, 12 canaux et
1 920 échantillons ; ses paramètres et la version exacte de la source sont dans
[`results/osf_backbone_compatibility.json`](../results/osf_backbone_compatibility.json).

**Limite :** cette vérification utilise des poids aléatoires. Elle établit une
compatibilité de code sur les cas testés, pas une reproduction des performances
OSF ni une validation sur patients.

## Checkpoint préentraîné chargé le 10 novembre 2026

Le fichier publié `osf_backbone.pth` (licence MIT, accès libre, révision Hugging Face
`f2067f2`, SHA-256 `c51190b1…31857`, conforme à l'empreinte publiée) se charge dans
notre backbone **en mode strict** : 0 clé manquante, 0 inattendue, 85 325 568 paramètres.
Métadonnées : DINO, ViT-Base, 12 canaux, 64 Hz, fenêtres de 30 s (1 920 échantillons),
découpage par groupes de canaux (`lead_wise=1`, groupes de 4). Un passage avant sur un
tenseur synthétique donne des sorties finies (CLS 768, 90 tokens). Le code
`osf/backbone/vit1d_cls.py` publié avec ces poids a la même empreinte git (`ad070eb6`)
que la source comparée à notre implémentation. Rapport :
[`results/osf_checkpoint_check.json`](../results/osf_checkpoint_check.json).

```bash
mkdir -p third_party/osf
curl -L -o third_party/osf/osf_backbone.pth \
    https://huggingface.co/yang-ai-lab/OSF-Base/resolve/f2067f2d95b3bff43e4a690a471ba3d4457f12a7/osf_backbone.pth
python scripts/check_osf_checkpoint.py third_party/osf/osf_backbone.pth --report results/osf_checkpoint_check.json
```

**Équivalence avec le code officiel, vrais poids** : le code officiel (même révision)
et notre backbone chargent ces poids sans clé manquante ni inattendue et donnent des
sorties **identiques** (écart maximal 0 sur le CLS et les 90 tokens, entrées synthétiques
d'écart-type 1 et 50) :
[`results/osf_real_weights_equivalence.json`](../results/osf_real_weights_equivalence.json).

**Non fait :** toute évaluation sur patients. Le mapping des canaux PhysioNet 2018 vers les 12 canaux OSF reste à écrire.

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

### Contrôler des époques déjà préparées

`src/data/osf_input.py` vérifie le format et remet les canaux dans l'ordre
canonique du [README officiel](https://github.com/yang-ai-lab/OSF-Open-Sleep-FM#input-format).
L'entrée doit déjà être flottante, à 64 Hz, avec 1 920 échantillons par fenêtre.
Le résultat est une copie contiguë float32, accompagnée de l'ordre source et
de la correspondance appliquée. Aucun rééchantillonnage, normalisation d'amplitude
ou remplissage de canal manquant n'est effectué.

La commande suivante contrôle un NPZ préparé séparément, contenant `epochs`
de forme `(n, canaux, 1920)`, `channels` comme vecteur Unicode et `sample_rate`
comme scalaire :

```bash
python scripts/check_osf_input.py epochs_osf.npz --report results/osf_input_check.json
```

Le rapport conserve l'empreinte du NPZ et le détail des canaux ; aucun modèle
n'est chargé et aucun signal corrigé n'est écrit. Les noms différents exigent
un fichier JSON passé via `--mapping`, de la forme `{nom OSF: nom source}`.
Un même canal source ne peut pas remplacer plusieurs canaux OSF.

**Limite :** un format valide ne prouve ni les unités, ni les références des
électrodes, ni la conformité au prétraitement SleepBench. Ces points restent
à valider pour chaque cohorte. Cet outil ne fournit donc pas de mapping
physiologique PhysioNet/NSRR et ne rend pas encore la comparaison exécutable.

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

Les jeux SleepBench nécessitent les accès aux cohortes correspondantes ; le
[README officiel](https://github.com/yang-ai-lab/OSF-Open-Sleep-FM#-sleepbench)
renvoie au NSRR et annonce encore les scripts de prétraitement pour une version
ultérieure. L'accès aux données et leur harmonisation restent donc à établir.

1. Écrire le mapping de canaux PhysioNet 2018 -> format d'entrée OSF (12ch, 64Hz)
2. Choisir le schéma de labels commun (4 classes, en fusionnant N1+N2 côté SleepFM)
3. Évaluer les deux sur exactement les mêmes epochs de test

## Proposition de protocole (10 novembre 2026, à valider avec le tuteur)

Question : sur les mêmes nuits et les mêmes fenêtres, les embeddings figés d'OSF
stadifient-ils mieux ou moins bien que ceux de SleepFM ?

### Canaux PhysioNet 2018 face aux 12 canaux d'OSF

OSF découpe les canaux par groupes de 4, dans cet ordre : l'ordre compte.

| Groupe OSF | Canal OSF | PhysioNet 2018 | Statut |
|---|---|---|---|
| 1 | ECG | ECG | Présent |
| 1 | EMG_Chin | Chin1-Chin2 | Présent |
| 1 | EMG_LLeg, EMG_RLeg | — | **Absents** |
| 2 | ABD | ABD | Présent |
| 2 | THX | CHEST | Présent (même rôle) |
| 2 | NP (pression nasale) | AIRFLOW | **À vérifier** : capteur peut-être différent |
| 2 | SN (ronflement) | — | **Absent** |
| 3 | EOG_E1_A2 | E1-M2 | Présent (M2 = mastoïde, comme A2) |
| 3 | EOG_E2_A1 | — | **Absent** |
| 3 | EEG_C3_A2, EEG_C4_A1 | C3-M2, C4-M1 | Présents |

SaO2 et les EEG frontaux et occipitaux de PhysioNet 2018 n'ont pas de place chez OSF.
4 canaux sur 12 manquent, et un est incertain. L'article d'OSF montre que les modèles
fondamentaux généralisent mal aux canaux manquants : **sur PhysioNet 2018, la
comparaison désavantagerait OSF**.

Autres inconnues : le prétraitement de SleepBench (unités, normalisation d'amplitude,
filtrage) n'est pas encore publié, et la manière dont OSF attend les canaux absents
(zéros, ou autre) n'est pas documentée.

### Deux options

| | A. Exploratoire, tout de suite | B. Comparaison équitable, après NSRR |
|---|---|---|
| Données | PhysioNet 2018 : les 500 sujets de développement, découpage figé (400 / 100) | MESA ou SHHS, cohortes de SleepBench |
| Canaux OSF | 8 sur 12, absents remplis de zéros, rééchantillonnage à 64 Hz | Montage proche de celui d'OSF |
| Prérequis | Script d'entrée OSF, run Kaggle d'embeddings (environ 1 h) | Accès NSRR ; idéalement le prétraitement de SleepBench publié |
| Ce que ça dit | Un premier ordre de grandeur, biaisé contre OSF | Une vraie comparaison des deux modèles |
| Statut du résultat | Exploratoire, jamais présenté comme un classement | Comparaison principale, critère fixé avant |

### Règles communes, quelle que soit l'option

1. Mêmes nuits et mêmes fenêtres de 30 s pour les deux modèles ; embeddings figés.
2. Mêmes sondes (logistique pondérée et BiGRU), entraînées sur les mêmes nuits d'entraînement, mêmes réglages.
3. Deux schémas de stades : 5 classes, et 4 classes comme dans l'article d'OSF (éveil, sommeil léger = N1 + N2, profond = N3, REM).
4. Critère fixé avant le calcul : écart de κ apparié par sujet (bootstrap, 1 000 tirages, graine 42), avec IC à 95 %.
5. Nuits de test sans aucun choix de réglage ; résultat publié quel qu'il soit.

**Recommandation : faire A maintenant, en exploratoire, pour avoir le pipeline OSF
prêt et un premier chiffre ; réserver la conclusion à B.** Les 494 sujets réservés,
déjà consultés, ne servent pas ici.

### Option A : code prêt (10 novembre 2026), run non lancé

- `src/data/osf_physionet2018.py` : part des époques annotées extraites pour SleepFM
  (mêmes fenêtres), rééchantillonne à 64 Hz, place les 8 canaux disponibles, met les
  4 autres à zéro (comme le code officiel pour un canal absent), z-score chaque canal
  sur la nuit puis borne à ±6. Le bornage vient du code officiel ; la normalisation
  par nuit est **une hypothèse**, le prétraitement de SleepBench n'étant pas publié.
- `scripts/embed_osf_physionet2018.py` et le notebook `sleepfm_osf_physionet2018.ipynb` :
  embeddings CLS (768 valeurs) des 500 sujets de développement, avec la même reprise
  et le même préchargement que les autres runs. Les poids sont vérifiés par leur SHA-256.
- Reste à écrire avant tout chiffre : les sondes sur embeddings de 768 valeurs et la
  comparaison appariée, avec le critère fixé avant le calcul.

### Option A : critère fixé avant le calcul (10 novembre 2026)

Écrit et commité avant l'ouverture des embeddings OSF du run Kaggle (`a42e992`, 500 sujets,
0 rejet). Commande : `python scripts/compare_osf_sleepfm.py --sleepfm-dir embeddings/physionet2018_v2
--osf-dir embeddings/physionet2018_osf --out results/osf_vs_sleepfm_physionet2018`.

- **Données** : les 400 nuits d'entraînement et les 100 nuits de test du découpage figé ; mêmes
  fenêtres, mêmes stades et mêmes temps pour les deux modèles, sinon arrêt. Run OSF avec rejet refusé.
- **Sondes** : logistique pondérée et BiGRU, réentraînées par le même code pour les deux modèles,
  graine 42, réglages par défaut. Aucun réglage choisi sur le test.
- **Critère principal** : écart de κ **OSF moins SleepFM**, BiGRU, 5 classes, bootstrap apparié par
  sujet, 1 000 tirages, graine 42, IC à 95 %.
- **Lecture** : IC au-dessus de 0, OSF au-dessus ; en dessous, SleepFM au-dessus ; contenant 0, pas
  de différence établie. Dans tous les cas, **résultat exploratoire** : OSF tourne ici sans 4 de ses
  12 canaux et avec une normalisation supposée. Un SleepFM au-dessus ne montre pas qu'OSF est moins
  bon en général.
- **Secondaires** : même écart pour la logistique ; 4 classes (N1 et N2 fusionnés, comme l'article
  d'OSF, à partir des prédictions à 5 classes) ; F1 macro et F1 par stade.
- **Contrôle** : les sondes SleepFM réentraînées doivent retrouver l'ordre de grandeur du run à
  500 sujets (κ 0,599 et 0,664) ; un écart notable serait signalé.
- Résultat publié quel qu'il soit.
