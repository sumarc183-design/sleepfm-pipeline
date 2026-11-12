# Erreurs respiratoires SleepFM / OSF — 12 novembre 2026

**Le meilleur score global d’OSF ne se retrouve pas sur tous les types d’événements.** Les hypopnées sont la principale limite : elles concentrent **3 185 des 4 319 fenêtres positives manquées par OSF (73,7 %)**.

Analyse descriptive des **100 nuits et 85 230 fenêtres de test**, dont 11 438 positives. Modèles et seuils inchangés ; aucun entraînement ni accès NSRR. [Plan initial](../plans/Plan%20analyse%20erreurs%20respiratoires%20SleepFM%20OSF%2012-11.md) · [Détail et preuves](../../results/respiratory_errors_physionet2018.json).

## Par type d’événement

Les groupes sont exclusifs et totalisent exactement les 11 438 fenêtres positives. Une fenêtre contenant plusieurs types, même brièvement, est dans « plusieurs types ». **Rappel par fenêtre**, pas par événement complet ni par patient.

| Type | Positives | Nuits | Rappel SleepFM | Rappel OSF | Écart en points [IC 95 %] |
|---|---:|---:|---:|---:|---|
| Apnée obstructive seule | 2 724 | 85 | 66,5 % | 78,2 % | +11,7 [+6,4 ; +17,3] |
| Apnée centrale seule | 1 511 | 78 | 57,2 % | 73,9 % | +16,7 [+6,5 ; +27,9] |
| Apnée mixte seule | 362 | 25 | 85,1 % | 94,8 % | +9,7 [+2,8 ; +25,7] |
| Hypopnée seule | 6 238 | 96 | 54,1 % | 48,9 % | -5,2 [-12,2 ; +2,7] |
| Plusieurs types | 603 | 64 | 79,6 % | 79,3 % | -0,3 [-8,7 ; +7,2] |

Les IC sont descriptifs : 1 000 tirages appariés par sujet, sans correction des comparaisons multiples. Les seuils sont différents, chacun choisi sur validation ; ces rappels ne comparent pas les modèles à spécificité égale. Les apnées mixtes ne concernent que 25 nuits.

**Hypopnées : aucun gain établi.** Le rappel mesuré passe de 54,1 % à 48,9 % ; l’IC de l’écart contient zéro. Cela ne démontre ni une baisse certaine ni l’équivalence des modèles. L’avantage sur les apnées obstructives et centrales décrit les pipelines aux seuils retenus, sans identifier sa cause.

## Selon le stade expert

| Stade | Positives | Rappel SleepFM | Rappel OSF | Faux positifs OSF sur fenêtres négatives |
|---|---:|---:|---:|---:|
| Éveil | 132 | 30,3 % | 40,2 % | 708 / 17 854 (4,0 %) |
| N1 | 3 632 | 59,9 % | 67,4 % | 1 359 / 9 924 (13,7 %) |
| N2 | 5 723 | 62,4 % | 63,0 % | 1 828 / 27 830 (6,6 %) |
| N3 | 471 | 45,2 % | 38,4 % | 127 / 9 473 (1,3 %) |
| REM | 1 480 | 56,7 % | 56,0 % | 755 / 8 711 (8,7 %) |

Les faux positifs sont proportionnellement les plus fréquents en N1. Le rappel OSF reste faible en N3 ; ces observations ne séparent pas les effets du stade, du type d’événement et du patient.

## Nuits à examiner

Cinq rappels OSF les plus bas parmi les nuits ayant au moins dix fenêtres positives, selon la règle fixée avant le regroupement. Cette sélection vise les erreurs et ne représente pas la moyenne du test.

| Nuit | Positives | Manquées par OSF | Rappel OSF |
|---|---:|---:|---:|
| tr04-0223 | 11 | 11 | 0,0 % |
| tr04-1006 | 38 | 38 | 0,0 % |
| tr06-0193 | 305 | 302 | 1,0 % |
| tr03-0592 | 35 | 34 | 2,9 % |
| tr06-0677 | 10 | 9 | 10,0 % |

[Ouvrir le cas tr06-0193](../../results/demo_errors_tr06-0193.html) : parmi ces cinq nuits, c’est celle qui contient le plus de fenêtres positives. **Choix d’illustration effectué après l’analyse**, pour lire un échec marqué, pas pour présenter une performance représentative. Aucune cause de cet échec n’est établie ; aucun signal brut n’a été réexaminé.

## À faire

- Examiner les fenêtres d’hypopnée et les nuits difficiles avec le tuteur ou le médecin, en conservant les seuils actuels.
- Choisir la cible exacte et l’usage avant un nouveau développement. Tout réglage se fera sur train/validation et demandera une évaluation distincte.
- Poursuivre la recherche d’une autre cohorte selon les accès autorisés. Les résultats ci-dessus ne valident ni un diagnostic ni un IAH.

Les tableaux complets par nuit et les groupes par fenêtre sont conservés dans `sleepfm-pipeline/results/`. Effectifs et erreurs retrouvent exactement le résultat global. Les IC ont été vérifiés indépendamment par copie explicite des nuits tirées : écart maximal 0. Empreintes des sources archivées et des résultats concordantes.

**Vérifications :** 274 tests rapides réussis, 2 tests XGBoost sautés sur Mac, 7 tests lents non relancés ; lint et formatage réussis. Le parcours de la nuit difficile est vérifié en DOM JavaScript sur trois étendues, sans attestation du rendu visuel du navigateur.
