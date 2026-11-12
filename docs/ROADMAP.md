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
  - [x] Confirmation sur les 494 sujets jamais ouverts (2026-11-09) : protocole validé par le tuteur, passage unique, gain du BiGRU confirmé (écart de κ +0,094 [+0,086 ; +0,103]) ; [résultat](PROTOCOLE_CONFIRMATION.md#résultat-2026-11-09-passage-unique). Les 994 sujets annotés de PhysioNet 2018 sont désormais tous utilisés
- [x] Demande d'accès NSRR (MESA, SHHS) déposée le 2026-11-04, réponse attendue (voir DATA_ACCESS.md) : évaluation supplémentaire, pas un préalable à la comparaison à l'article

## J60 — Comparaison
- [ ] Embeddings + sonde linéaire sur MESA (cohorte supplémentaire, si l'accès NSRR est obtenu)
- [ ] Comparaison SleepFM vs OSF sur SleepBench (même protocole, mêmes données)
  - [x] Comparaison exploratoire sur PhysioNet 2018 (option A, critère fixé avant) : OSF au-dessus, BiGRU κ 0,717 contre 0,664, écart +0,052 [+0,032 ; +0,073], avec 4 canaux OSF absents (`results/osf_vs_sleepfm_physionet2018.json`). À confirmer sur MESA ou SHHS
  - [x] Compatibilité du backbone avec le code officiel : noms de poids, deux modes de découpage, vérification indépendante sur tenseurs synthétiques. Checkpoint préentraîné et comparaison sur patients non exécutés (`results/osf_backbone_compatibility.json`)
  - [x] Checkpoint préentraîné OSF chargé en mode strict (0 clé manquante ou inattendue), passage avant synthétique fini (`results/osf_checkpoint_check.json`), sorties identiques au code officiel avec ces poids (`results/osf_real_weights_equivalence.json`). Mapping des canaux PhysioNet 2018 et protocole de comparaison restent à faire
  - [x] Contrôle du format des entrées OSF : canaux, ordre explicite, 64 Hz, fenêtres de 30 s et valeurs finies ; tests synthétiques (`scripts/check_osf_input.py`). Unités, références d'électrodes et mapping de cohorte restent à valider
- [ ] Tableau de résultats consolidé pour la comparaison multi-cohortes
  - [x] Consolidation des résultats PhysioNet 2018 déjà enregistrés (100, 200, 500 sujets et confirmation) en CSV et Markdown, avec sources et empreintes ; aucune nouvelle évaluation (`scripts/consolidate_results.py`)
- [x] Prédiction sur une polysomnographie brute non annotée au format PhysioNet 2018 : fenêtres fixes de 30 s indépendantes des annotations, contrat du signal vérifié, traitement après enregistrement (`scripts/predict_raw_night.py`). Première évaluation uniforme le 10 novembre (`scripts/evaluate_raw_grid.py`), complétée le 12 novembre par majorité stricte (`scripts/evaluate_grid_transitions.py`) : cas à un seul échantillon minoritaire séparés, indéfini exclu, anciens scores retrouvés. Résultats et limites dans [RESULTATS.md](RESULTATS.md#prédiction-sur-psg-brute-grille-fixe).

## J90 — Différenciation
- [x] Incertitude du choix des seuils sur validation : bootstrap avec sélection
  et évaluation des nuits absentes, puis validation croisée par nuit ; compte
  rendu dans `Journal`. Confirmation externe et stabilité du transfert ouvertes.
- [x] Premier diagnostic par événements sur les seules nuits de validation :
  couverture des annotations, recouvrement positif et appariement un à un
  aux seuils existants ; compte rendu daté dans `Journal`. Règle d'alerte,
  évaluation complète et confirmation indépendante restent à définir.
- [x] Regroupement descriptif des fenêtres respiratoires : séquences, lacunes,
  durées et charge par heure évaluée, aux seuils déjà fixés ; [compte rendu
  du 12 novembre](resultats/respiratoire_sequences_2026-11-12.md). Évaluation événementielle encore distincte.
- [x] Compromis sensibilité/fausses fenêtres positives respiratoires : seuils
  choisis uniquement sur validation, modèles inchangés, rappel des hypopnées
  et IC par nuit ; [compte rendu du 12 novembre](resultats/respiratoire_seuils_2026-11-12.md).
- [x] Analyse descriptive des frontières de stade sur la grille brute : position
  de l'échantillon minoritaire, stades voisins et erreurs des sondes figées ;
  annotations et prédictions contrôlées, [compte rendu du 12 novembre](resultats/frontieres_stade_2026-11-12.md).
- [x] Première exploration apnées/hypopnées sur PhysioNet 2018 (12 novembre) :
  logistique sur embeddings figés et démo d'une nuit. [Compte rendu](resultats/respiratoire_baseline_osf_2026-11-12.md). La cible produit reste à choisir.
- [x] Comparaison respiratoire exploratoire, mêmes fenêtres et partition 360/40/100 :
  réalisée le 12 novembre ; démo comparative et [compte rendu](resultats/respiratoire_baseline_osf_2026-11-12.md).
- [x] Des fenêtres aux événements et à l'IAH (IAH ≥ 15 par nuit, [résultat](resultats/respiratoire_iah_2026-11-12.md)) : [plan du 12 novembre](plans/Plan%20%C3%A9v%C3%A9nements%20et%20IAH%20SleepFM%20OSF%2012-11.md),
  validé par le tuteur le 12 novembre ; confirmation sur NSRR, pas sur les 494 sujets
- [x] Choix de la tâche différenciante : apnée, les 3 cibles de l'étude de faisabilité (tuteur, 12 novembre)
- [ ] Chaîne NSRR prête avant l'accès (lecture MESA/SHHS, annotations XML, IAH), pour appliquer
  la règle IAH figée dès l'ouverture des données (décision du tuteur, 12 novembre)
- [ ] Baseline mesurée sur cette tâche
- [ ] Note de synthèse pour décision sur la phase 2
