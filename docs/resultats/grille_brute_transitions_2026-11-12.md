# SleepFM — fenêtres mixtes sur grille PSG brute, 12 novembre 2026

**Résultat essentiel :** en réintégrant les fenêtres entièrement annotées par
majorité stricte, le κ du BiGRU passe de **0,711 à 0,664**. Les fenêtres
réintégrées ont toutes un seul échantillon minoritaire (5 ms). Cela ne démontre
pas une difficulté sur des transitions cliniques de plusieurs secondes.

## Périmètre et règle

100 nuits de test PhysioNet 2018 déjà examinées, 91 947 fenêtres fixes de 30 s.
Même grille brute, mêmes embeddings et mêmes sondes figées du run à 500 sujets ;
aucun entraînement ni réglage. Le BiGRU conserve le contexte de toute la nuit,
y compris les fenêtres exclues du score. Ni données NSRR ni nouvelle utilisation
des 494 sujets de confirmation.

La référence complémentaire attribue le stade occupant strictement plus de
15 s, seulement si toute la fenêtre est annotée. Il s'agit d'une règle
d'évaluation, pas d'une nouvelle cotation clinique.

| Catégorie | Fenêtres | Traitement |
|---|---:|---|
| Un seul stade, entièrement uniforme | 76 626 | Notées ; ancienne référence |
| Un seul échantillon minoritaire | 12 490 | Notées par majorité ; groupe séparé |
| Majorité avec plus d'un échantillon minoritaire | 0 | Aucun score disponible |
| Entièrement annotées, sans majorité stricte | 0 | Aucun score disponible |
| Entièrement indéfinies | 2 715 | Exclues |
| Partiellement indéfinies | 116 | Exclues |
| **Total** | **91 947** | **89 116 notées, 2 831 exclues** |

## Scores

IC à 95 % : bootstrap par nuit, 500 tirages, graine 42.

| Fenêtres | Logistique κ [IC 95 %] | F1 macro | BiGRU κ [IC 95 %] | F1 macro |
|---|---|---:|---|---:|
| Uniformes seules | 0,634 [0,603 ; 0,664] | 0,707 | 0,711 [0,678 ; 0,742] | 0,759 |
| Un échantillon minoritaire | 0,369 [0,346 ; 0,391] | 0,512 | 0,367 [0,346 ; 0,388] | 0,518 |
| **Ensemble notable par majorité** | **0,600 [0,570 ; 0,627]** | **0,686** | **0,664 [0,634 ; 0,694]** | **0,731** |

Le score baisse parce que l'ensemble évalué inclut des fenêtres plus difficiles ;
les modèles n'ont pas changé. Sur le groupe à un échantillon minoritaire, les
κ des deux sondes sont voisins : aucun avantage statistique n'a été testé
spécifiquement sur ce groupe. La convention observée ne prouve ni une erreur
d'annotation ni un problème de signal.

## Contrôles et changement de source avant les scores

Le [plan initial](../plans/Plan%20transitions%20PSG%20brute%20SleepFM%2012-11.md)
utilisait les marqueurs WFDB. Le contrôle préalable a arrêté le calcul sur
tr03-0052 : 84 labels uniformes différaient. L'examen a révélé un écart d'un
échantillon entre ces marqueurs et les vecteurs HDF5 utilisés par l'évaluation
initiale. Aucun nouveau score n'avait été calculé.

Le [plan v2](../plans/Plan%20transitions%20PSG%20brute%20SleepFM%2012-11%20v2.md),
fixé avant les scores, reprend exactement la source HDF5 et sépare les cas
à un échantillon minoritaire. Seules les petites annotations publiques des
100 nuits ont été téléchargées, sans signaux PSG.

- Labels uniformes retrouvés exactement sur les **100 nuits** ; κ et F1
  des deux sondes identiques à l'évaluation initiale.
- Recalcul indépendant des métriques et matrices depuis les prédictions ;
  écart maximal des IC recalculés : **1,7 × 10⁻¹⁶**. Reconstruction indépendante
  des majorités, échantillon par échantillon, sur trois nuits : identique.
- Empreintes des caches et annotations HDF5 des 100 nuits, probabilités,
  archive du code et plan vérifiés. En-têtes et annotations WFDB contrôlés
  contre leur manifeste lors du calcul.
- **285 tests réussis, 2 ignorés, 7 tests lents désélectionnés**. Les tests
  lents n'ont pas été exécutés ; aucune nouvelle validation clinique.

Révision de base : `59efe4b106124f265586dbedafd53ed3f757adef`, avec modifications
locales non commitées. L'archive conserve le code utilisé et le plan.

## Preuves conservées

- [Résultat et métriques détaillées](../../results/raw_grid_transitions_physionet2018.json)
- [Labels, groupes et prédictions](../../results/raw_grid_transitions_physionet2018.predictions.npz)
- [Inventaire et empreintes des entrées](../../results/raw_grid_transitions_physionet2018.inputs.json)
- [Archive du code et du plan](../../results/raw_grid_transitions_physionet2018.code.zip)
- [Arrêt préalable du plan initial](../../results/raw_grid_transitions_precheck.json)
- [Évaluation uniforme initiale](../../results/raw_grid_physionet2018.json)

## À faire

Complément réalisé le même jour : [analyse des frontières de stade](frontieres_stade_2026-11-12.md), position de l'échantillon minoritaire et répartition des erreurs.

- Présenter l'ensemble notable par majorité avec la mesure uniforme et leurs
  périmètres ; ne plus décrire toutes les exclusions comme des transitions
  de plusieurs secondes.
- Faire examiner quelques frontières de stade par le tuteur ou le médecin
  avant de décider d'une autre règle de cotation.
- Confirmer le transfert sur une autre cohorte dès l'accès NSRR ; les nuits
  présentes sont déjà explorées, sans confirmation indépendante.
