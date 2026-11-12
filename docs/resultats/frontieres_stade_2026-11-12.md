# SleepFM — frontières de stade, 12 novembre 2026

**Les 12 490 fenêtres réintégrées ont leur échantillon minoritaire à la toute
fin des 30 secondes.** Dans 12 488 cas, ce stade est celui de la fenêtre suivante ;
les deux autres sont les dernières fenêtres complètes de leur nuit. Cela précise la
convention d'annotation, sans prouver une erreur de cotation ou une anticipation
du modèle.

Analyse descriptive des mêmes 100 nuits déjà examinées : prédictions archivées,
aucun entraînement, seuil ni modèle changé. Pas de nouvelle donnée téléchargée,
de signal relu, d'utilisation de NSRR ou des 494 sujets de confirmation.

## Vers quel stade se trompent les modèles ?

Sur les 12 490 fenêtres à un échantillon minoritaire :

| Prédiction | Logistique, % [IC 95 %] | BiGRU, % [IC 95 %] |
|---|---:|---:|
| Stade majoritaire : correcte sous la règle retenue | 51,9 [50,1 ; 53,7] | 53,1 [51,4 ; 54,6] |
| Stade de l'unique échantillon minoritaire | 32,8 [31,3 ; 34,2] | 36,2 [35,0 ; 37,4] |
| Un autre stade | 15,3 [13,4 ; 17,4] | 10,7 [9,0 ; 12,7] |

IC par bootstrap de nuits, 500 tirages, graine 42. Le stade minoritaire
représente **4 523 des 5 864 erreurs du BiGRU** dans ce groupe. Le BiGRU lit
la nuit entière : ce résultat ne constitue pas une mesure causale de son
comportement temporel. Aucun test de supériorité n'a été effectué ici.

## Tous les stades ne sont pas touchés de la même façon

| Stade majoritaire | Fenêtres uniformes | Un échantillon minoritaire | Rappel logistique sur ce groupe | Rappel BiGRU sur ce groupe |
|---|---:|---:|---:|---:|
| Éveil | 15 758 | 2 480 | 60,0 % | 57,9 % |
| N1 | 10 401 | 4 358 | 47,3 % | 38,2 % |
| N2 | 31 471 | 3 771 | 44,9 % | 69,7 % |
| N3 | 8 936 | 1 197 | 67,9 % | 40,9 % |
| REM | 10 060 | 684 | 62,1 % | 59,6 % |

N1 représente **34,9 %** du groupe à un échantillon minoritaire, contre
**13,6 %** des fenêtres uniformes : l'exclusion modifiait aussi la composition
des stades évalués. Ces rappels sont descriptifs, sans comparaison statistique
par stade.

La paire la plus fréquente est **N1 majoritaire / N2 minoritaire** : 2 499
fenêtres, présentes dans les 100 nuits. Le BiGRU prédit N1 dans 39,7 % des cas
et N2 dans 52,2 %. Toutes les paires sont conservées dans le JSON ; les petites
paires ne justifient pas une interprétation détaillée.

## Vérifications et preuves

- [Plan fixé avant le calcul](../plans/Plan%20fronti%C3%A8res%20de%20stade%20SleepFM%2012-11.md).
- Labels majoritaires et catégories reconstruits depuis les HDF5 sur les
  **100 nuits** : identiques à la référence. Matrices des deux sondes retrouvées
  sur fenêtres uniformes, à un échantillon minoritaire et ensemble notable.
- Empreintes des annotations, prédictions, inventaire et code contrôlées.
  Recalcul indépendant des IC en rééchantillonnant explicitement les lignes
  par nuit : écart **nul** avec le résultat archivé.
- Tests ciblés : **20 réussis**, comprenant les protections contre une source
  modifiée, une reconstruction différente et l'écrasement d'un résultat.
- Suite complète hors tests lents : **294 réussis, 2 ignorés, 7 désélectionnés**
  en 182,94 s. Les sept tests lents n'ont pas été exécutés. Lint, formatage
  et contrôle des espaces du diff réussis.

Révision de base : `59efe4b106124f265586dbedafd53ed3f757adef`, avec modifications
locales non commitées. La procédure archive son code, le lecteur de labels et le plan.

- [Résultat détaillé, paires et IC](../../results/stage_boundaries_physionet2018.json)
- [Groupes, stades voisins et prédictions](../../results/stage_boundaries_physionet2018.groups.npz)
- [Archive des sources et du plan](../../results/stage_boundaries_physionet2018.code.zip)
- [Évaluation précédente et ses limites](grille_brute_transitions_2026-11-12.md)

## À faire

- Examiner quelques frontières N1/N2 et N3/N2 avec le tuteur ou le médecin,
  en confrontant annotations et signaux avant de choisir une autre règle.
- Conserver les résultats actuels comme référence ; tester tout changement
  de méthode sur la validation, puis le confirmer sur des données indépendantes.
- Pour comparer OSF sur PSG brute, produire une extraction OSF distincte :
  le cache actuel couvre la grille annotée, légèrement décalée. Aucune nouvelle
  performance OSF sur grille brute n'est annoncée par cette analyse.
