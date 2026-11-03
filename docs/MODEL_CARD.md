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

## Usage prévu

- Recherche et évaluation interne : vérifier qu'on reproduit le comportement
  publié de SleepFM et mesurer ce qu'apportent d'autres sondes.
- **Hors périmètre** : tout usage clinique, diagnostic ou aide à la décision
  médicale. Ce n'est pas un dispositif médical et il n'a été ni validé
  cliniquement ni évalué par un comité.

## Données d'évaluation

- **PhysioNet/CinC Challenge 2018** (accès libre) : polysomnographies adultes
  annotées par des experts (stades AASM), 13 canaux rééchantillonnés à 256 Hz.
  Les 100 premiers sujets d'entraînement du challenge **par identifiant**
  (pas un tirage aléatoire parmi les 994) ont été utilisés.
- **Split par sujet** (jamais par époque) : 80/20 par défaut (75/25 pour les
  résultats publiés au 2026-11-03). Les sondes non linéaires sont départagées
  par validation croisée par sujet sur le train ; le test n'est utilisé
  qu'une fois.

## Métriques

Accuracy, kappa de Cohen (accord corrigé du hasard, métrique de référence en
stadification), F1 macro, F1 par stade, ROC-AUC un-contre-tous, gap
train − test. Résultats à jour : [README](../README.md#résultats).

## Limites connues

- **Pas encore comparé au chiffre publié** : SleepFM rapporte ses résultats
  sur MESA/SHHS, pas sur PhysioNet 2018. La reproduction stricte attend
  l'accès NSRR.
- **N1 mal reconnu** (F1 ≈ 0,43) : stade de transition, rare et peu
  concordant même entre experts.
- **Population** : adultes d'un seul centre (PhysioNet 2018). Aucune
  évaluation en pédiatrie (CHAT, phase 2), ni sur d'autres appareils ou
  montages.
- **Canaux** : le montage 13 canaux est requis ; un enregistrement auquel il
  manque un canal est rejeté à l'extraction.
- **Variabilité** : avec 20 à 25 sujets de test, les chiffres peuvent varier
  de quelques centièmes selon le tirage des sujets ; d'où les intervalles de
  confiance par bootstrap sur les sujets.
- **Résultats du 2026-11-02/03** obtenus avec une régression logistique non
  pondérée, alors que le protocole officiel pondère les classes : à remplacer
  par le run suivant.

## Considérations éthiques

Données dé-identifiées en accès public. Pour MESA, SHHS et CHAT (NSRR),
respecter le Data Use Agreement : pas de tentative de ré-identification, pas
de redistribution, et vérifier avant tout calcul dans le cloud que l'accord
l'autorise.
