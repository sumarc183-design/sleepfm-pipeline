# Roadmap SleepFM

## J30 — Reproduction
- [x] Pipeline officiel SleepFM opérationnel sur PhysioNet Challenge 2018
- [x] Premier κ de stadification au protocole officiel (régression pondérée) : 0,607 [IC 95 % 0,545 ; 0,669] sur 100 sujets PhysioNet 2018, 20 sujets de test (2026-11-04) ; BiGRU sur la nuit 0,685 (0,666 à 0,684 sur 3 graines). Premiers essais (non pondérés) : 0,584 puis 0,622
- [x] 200 sujets (160 / 40, découpage figé étendu, 2026-11-05) : baseline κ 0,587 [0,533 ; 0,643], BiGRU 0,656 ; sur les 20 anciennes nuits de test, 0,620 et 0,706 (mieux qu'à 100 sujets)
- [x] 500 sujets (400 / 100, 2026-11-05) : baseline κ 0,599 [0,570 ; 0,627], BiGRU 0,664 [0,634 ; 0,694] ; évaluation élargie, 40 des 100 nuits de test déjà examinées
- [x] Comparaison chiffrée à l'article SleepFM 2024, même ordre de grandeur (reproduction stricte non démontrée) (tableau 8, PhysioNet 2018 : ROC-AUC 0,924, AUPRC 0,759, F1 0,700, 100 participants en test ; pas de κ publié)
  - [x] Premier ordre de grandeur : ROC-AUC 0,925 et F1 0,683 avec la baseline officielle
  - [x] AUPRC (0,770 contre 0,759) et comparaison stade par stade : le profil du F1 de la logistique non pondérée colle à celui de l'article (hypothèse à vérifier)
  - [x] Même effectif de test (100 nuits), listes de sujets enregistrées (2026-11-05). Baseline ROC-AUC 0,921, AUPRC 0,750, F1 0,686 contre 0,924 / 0,759 / 0,700 ; BiGRU 0,940 / 0,800 / 0,731. Restent différents ou inconnus : sujets, découpage, sonde et poids exacts de l'article
  - [ ] Confirmation sur les 494 sujets jamais ouverts : [protocole](PROTOCOLE_CONFIRMATION.md) et commande prêts, répétés sur des nuits déjà ouvertes ; à valider avec le tuteur avant calcul. Elle complète aussi PhysioNet 2018 (500 sujets sur 994 pour l'instant)
- [x] Demande d'accès NSRR (MESA, SHHS) déposée le 2026-11-04, réponse attendue (voir DATA_ACCESS.md) : évaluation supplémentaire, pas un préalable à la comparaison à l'article

## J60 — Comparaison
- [ ] Embeddings + sonde linéaire sur MESA (cohorte supplémentaire, si l'accès NSRR est obtenu)
- [ ] Comparaison SleepFM vs OSF sur SleepBench (même protocole, mêmes données)
- [ ] Tableau de résultats consolidé

## J90 — Différenciation
- [ ] Choix de la tâche différenciante (détection apnée OU robustesse pédiatrique CHAT)
- [ ] Baseline mesurée sur cette tâche
- [ ] Note de synthèse pour décision d'investissement phase 2
