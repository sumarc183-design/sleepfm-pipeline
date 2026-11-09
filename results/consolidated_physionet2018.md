# Résultats consolidés enregistrés

Lecture des JSON sauvegardés, sans entraînement ni recalcul des scores. Les cohortes de test et les prétraitements peuvent différer : ces lignes ne forment pas un classement.

Un champ que le JSON n'enregistre pas est noté « absent ». La confirmation est conservée séparément des évaluations de développement et des répétitions.

| Source | Sonde | Nuits train | Nuits test | Époques test | κ [IC95 %] | F1 macro | ROC-AUC | AUPRC |
|---|---|---:|---:|---:|---|---:|---:|---:|
| R1 | logistic | 80 | 20 | 17818 | 0,607 [0,545 ; 0,669] | 0,683 | 0,925 | 0,770 |
| R1 | logistic_unweighted | 80 | 20 | 17818 | 0,645 [0,577 ; 0,702] | 0,697 | 0,927 | 0,771 |
| R1 | mlp | 80 | 20 | 17818 | 0,623 [0,556 ; 0,684] | 0,683 | 0,924 | 0,762 |
| R1 | sequence | 80 | 20 | 17818 | 0,685 [0,613 ; 0,740] | 0,739 | 0,936 | 0,801 |
| R2 | logistic | 160 | 40 | 36163 | 0,587 [0,533 ; 0,643] | 0,679 | 0,916 | 0,750 |
| R2 | logistic_unweighted | 160 | 40 | 36163 | 0,608 [0,544 ; 0,666] | 0,685 | 0,916 | 0,752 |
| R2 | mlp | 160 | 40 | 36163 | 0,586 [0,525 ; 0,646] | 0,660 | 0,909 | 0,725 |
| R2 | sequence | 160 | 40 | 36163 | 0,656 [0,594 ; 0,713] | 0,733 | 0,933 | 0,791 |
| R3 | logistic | 400 | 100 | 89132 | 0,599 [0,570 ; 0,627] | 0,686 | 0,921 | 0,750 |
| R3 | logistic_unweighted | 400 | 100 | 89132 | 0,613 [0,581 ; 0,641] | 0,688 | 0,921 | 0,751 |
| R3 | mlp | 400 | 100 | 89132 | 0,591 [0,558 ; 0,621] | 0,666 | 0,916 | 0,736 |
| R3 | sequence | 400 | 100 | 89132 | 0,664 [0,634 ; 0,694] | 0,731 | 0,940 | 0,800 |
| R4 | logistic | absent | 494 | 443289 | 0,578 [0,565 ; 0,592] | 0,671 | 0,916 | 0,742 |
| R4 | sequence | absent | 494 | 443289 | 0,672 [0,661 ; 0,683] | 0,739 | 0,942 | 0,811 |

## Sources

- R1 : `results/results_physionet2018_100_auprc.json` ; évaluation enregistrée ; prétraitement : absent ; graine : 42 ; SHA-256 `3535360fd68047fb373acd9891a0a86805f84162c1b75236bd37778c485e0885`.
- R2 : `results/results_physionet2018_200.json` ; évaluation enregistrée ; prétraitement : 2 ; graine : 42 ; SHA-256 `62ba62544511d2a05e65274b0ccb1731faf5c780a4c9c24676d6d88034048429`.
- R3 : `results/results_physionet2018_500.json` ; évaluation enregistrée ; prétraitement : 2 ; graine : 42 ; SHA-256 `c3de6fbab9ad082d3eb984c49c87feddf300d402abd79f6ede70c5a5c645a451`.
- R4 : `results/confirmation_physionet2018.json` ; confirmation sur les sujets réservés ; prétraitement : 2 ; graine : absent ; SHA-256 `ad190e34e146c993fe37745aa54249562d4971c63998604f4248fce8d115de68`.
  Le JSON de confirmation ne garde que les empreintes des modèles figés. D'après `docs/PROTOCOLE_CONFIRMATION.md`, ce sont les exports du run à 500 sujets (400 nuits d'entraînement, graine 42), sans réentraînement.
