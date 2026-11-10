# Carte du modèle — SleepFM Pipeline

Format inspiré des *Model Cards* (Mitchell et al., 2019).

## Description

- **Modèle de base** : SleepFM, checkpoint officiel `best.pt`
  ([rthapa84/sleepfm-codebase](https://github.com/rthapa84/sleepfm-codebase)),
  trois encodeurs EffNet 1D (respiratoire, stades, ECG) préentraînés de façon
  auto-supervisée (contrastive) sur des polysomnographies. **Figé** : aucun
  poids de SleepFM n'est modifié ici.
- **Ce qui est entraîné** : des sondes supervisées légères sur les embeddings
  concaténés (1 536 dimensions) : régression logistique pondérée (baseline,
  protocole officiel), régression logistique non pondérée, MLP ou XGBoost (un
  seul retenu), BiGRU sur la suite des époques d'une nuit.
- **Tâche** : stadification du sommeil par époque de 30 s, 5 classes
  (Wake, N1, N2, N3, REM).
- **Inférence sans annotations** : `scripts/predict_raw_night.py`, pour les fichiers
  PhysioNet 2018 `.hea`/`.mat`, utilise une grille fixe depuis le début de la nuit.
  Le traitement du signal et le checkpoint sont contrôlés. Mesuré le 10 novembre 2026
  sur les 100 nuits de test : sur les fenêtres d'un seul stade (83 % des fenêtres),
  pas de baisse par rapport à la grille annotée (κ 0,711 [0,678 ; 0,742] pour le BiGRU, 0,634 [0,603 ; 0,664] pour la baseline).
  Les transitions, à cheval sur deux stades, ne sont pas notées : score possiblement
  optimiste. Voir [RESULTATS.md](RESULTATS.md#prédiction-sur-psg-brute-grille-fixe).

## Usage prévu

- Recherche et évaluation interne : vérifier qu'on reproduit le comportement
  publié de SleepFM et mesurer ce qu'apportent d'autres sondes.
- **Hors périmètre** : tout usage clinique, diagnostic ou aide à la décision
  médicale. Ce n'est pas un dispositif médical et il n'a été ni validé
  cliniquement ni évalué par un comité.

## Données d'évaluation

- **PhysioNet/CinC Challenge 2018** (accès libre) : polysomnographies adultes
  annotées par des experts (stades AASM), 13 canaux rééchantillonnés à 256 Hz.
  Les 500 premiers sujets d'entraînement du challenge **par identifiant**
  (pas un tirage aléatoire parmi les 994) ont été utilisés.
- **Split par sujet** (jamais par époque), figé : 400 nuits d'entraînement
  et 100 de test (89 132 époques), comme l'article. Les nuits des runs à 100
  et 200 sujets gardent leur rôle. Sélection non refaite à 500 sujets (`selection: null` dans le JSON) ; aux runs précédents, les sondes non linéaires sont départagées
  par validation croisée par sujet sur le train ; le test ne sert à aucun
  choix, mais les 20 premières nuits de test ont été évaluées par plusieurs
  runs.

## Métriques

Accuracy, kappa de Cohen (accord corrigé du hasard, métrique de référence en
stadification), F1 macro, F1 par stade, ROC-AUC et AUPRC un-contre-tous, gap
train − test. Résultats à jour : [README](../README.md#résultats-en-bref).

## Limites connues

- **Pas une reproduction stricte de l'article** : l'article SleepFM 2024
  (section 4.6, tableau 8) évalue PhysioNet 2018 avec 100 participants en test
  (ROC-AUC 0,924, AUPRC 0,759, F1 0,700). La baseline obtient 0,921, 0,750 et
  0,686 sur 100 nuits : même effectif et même ordre de grandeur, mais sujets,
  découpage et sonde peuvent différer, et 40 de ces nuits avaient déjà été
  examinées. Confirmation réalisée le 9 novembre 2026 :
  [protocole et résultat](PROTOCOLE_CONFIRMATION.md#résultat-2026-11-09-passage-unique).
- **Sujets non tirés au hasard** : les 500 premiers par identifiant pour le
  développement ; confirmation unique sur les 494 autres, sans réentraînement :
  κ 0,672 pour le BiGRU et 0,578 pour la baseline. Cette confirmation reste
  dans la même cohorte. Toute analyse ultérieure est exploratoire ; ces sujets
  ne servent ni à choisir ni à régler une variante de stadification.
- **N1 mal reconnu** (F1 0,50 avec la baseline et le BiGRU, 0,45 sans
  pondération, run à 500 sujets), confondu surtout avec l'éveil : stade de transition, rare et peu
  concordant même entre experts.
- **Population** : adultes d'un seul centre (PhysioNet 2018). Aucune
  évaluation en pédiatrie (CHAT, phase 2), ni sur d'autres appareils ou
  montages.
- **Canaux** : le montage 13 canaux est requis ; un enregistrement auquel il
  manque un canal est rejeté à l'extraction.
- **Variabilité** : avec 20 à 100 sujets de test, les chiffres peuvent varier
  de quelques centièmes selon le tirage des sujets ; d'où les intervalles de
  confiance par bootstrap sur les sujets.
- **Résultats** (2026-11-05, protocole officiel, run de référence à 500
  sujets) : κ 0,599 [0,570 ; 0,627] pour la baseline, 0,664 [0,634 ; 0,694]
  pour le BiGRU ; ROC-AUC 0,921, AUPRC 0,750, F1 0,686 pour la baseline,
  contre 0,924 / 0,759 / 0,700 dans l'article. Écart apparié du BiGRU : κ
  +0,065 [+0,047 ; +0,083], graine 42 ; sur 3 graines à 500 sujets, BiGRU de 0,664 à 0,682 et gain de +0,065 à +0,083. 5 nuits sur 100 sont
  sous κ 0,3 avec la baseline. Historique : à 200 sujets (2026-11-05), 0,587 et
  0,656 ; à 100 sujets en prétraitement version 1 (2026-11-04), 0,607 et 0,685
  (BiGRU de 0,666 à 0,684 sur 3 graines d'entraînement). Les résultats du
  2026-11-02/03 (κ 0,584 et 0,622) utilisaient une régression non pondérée.

## Considérations éthiques

Données dé-identifiées en accès public. Pour MESA, SHHS et CHAT (NSRR),
respecter le Data Use Agreement : pas de tentative de ré-identification, pas
de redistribution, et vérifier avant tout calcul dans le cloud que l'accord
l'autorise.
