# Roadmap SleepFM

## J30 — Reproduction
- [x] Pipeline officiel SleepFM opérationnel sur PhysioNet Challenge 2018
- [ ] Premier κ de stadification, comparé au chiffre publié (tolérance ±0.05)
  - κ au protocole officiel (régression pondérée) : 0,606 [IC 95 % 0,545 ; 0,669] sur 100 sujets PhysioNet 2018, 20 sujets de test (2026-11-04) ; BiGRU sur la nuit 0,684. Premiers essais (non pondérés) : 0,584 puis 0,622. Pas de chiffre publié sur PhysioNet 2018, la comparaison se fera sur MESA
- [ ] Demande d'accès NSRR/MESA déposée (voir DATA_ACCESS.md)

## J60 — Comparaison
- [ ] Embeddings + sonde linéaire sur MESA (reproduction de la démo officielle)
- [ ] Comparaison SleepFM vs OSF sur SleepBench (même protocole, mêmes données)
- [ ] Tableau de résultats consolidé

## J90 — Différenciation
- [ ] Choix de la tâche différenciante (détection apnée OU robustesse pédiatrique CHAT)
- [ ] Baseline mesurée sur cette tâche
- [ ] Note de synthèse pour décision d'investissement phase 2
