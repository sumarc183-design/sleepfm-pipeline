# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [Non publié]

### Corrigé (audit du 2026-11-09)
- **Modèles de la confirmation** : l'empreinte est vérifiée sur chaque fichier réellement chargé, et
  seules les deux sondes figées sont chargées. Un manifeste qui redirige une sonde vers un autre
  fichier est refusé avant toute trace et tout score. Le rapport enregistre les fichiers chargés et
  l'empreinte du manifeste. Export actuel rechargé sans changement.
- **Annotations invalides refusées** avant leur conversion en stades : colonnes attendues, indicateurs
  binaires finis, un seul stade actif par échantillon, longueurs cohérentes avec le signal et l'en-tête.
  Un échantillon sans stade actif ne devient plus N1. Les annotations réelles vérifiées sont conformes.
- **Rejets de la confirmation** : même règle que le bilan du run (raison lisible et non vide), et refus
  si aucune nuit n'est évaluable, avant tout score.
- **Comparaison entre runs** : mêmes fenêtres ne suffisent plus ; l'écart n'est attribué au seul
  entraînement que si les contrats des embeddings (JSON voisin de chaque archive) sont présents et
  identiques. Sinon il est présenté comme une comparaison de pipelines.
- **Fiche du modèle** : lien vers « Résultats en bref » réparé.

### Ajouté
- **Sondes figées versionnées** (`models/physionet2018_200` et `_500`, 15 Mo) : le dépôt cloné suffit
  pour relancer la confirmation ; empreintes identiques à celles de `results/confirmation_physionet2018.json`.
- **Bootstrap accéléré** (suggestion 4 de l'audit) : IC calculés depuis les matrices de confusion par sujet,
  mêmes tirages. Identiques sur données réelles (écart 0 sur la confirmation, 1e-16 sur le run à 500 sujets),
  environ 250 fois plus rapide (2 s au lieu de 500 s pour la confirmation).
- **Comparaison exploratoire SleepFM / OSF** (`scripts/compare_osf_sleepfm.py`, critère fixé avant le calcul) :
  sur les 100 nuits de test, mêmes fenêtres et mêmes sondes, OSF fait mieux (BiGRU κ 0,717 contre 0,664,
  écart +0,052 [+0,032 ; +0,073]) malgré 4 canaux absents. À confirmer sur MESA ou SHHS.
- **Entrée OSF depuis PhysioNet 2018** (option A, exploratoire) : `src/data/osf_physionet2018.py`
  (mêmes fenêtres que SleepFM, 8 canaux sur 12 et 4 mis à zéro, 64 Hz, z-score par nuit et bornage à ±6),
  `scripts/embed_osf_physionet2018.py` et notebook Kaggle `sleepfm_osf_physionet2018.ipynb`. Run non lancé ;
  sondes et comparaison restent à écrire.
- **Proposition de protocole SleepFM / OSF** (`docs/OSF_COMPARISON.md`) : correspondance des canaux
  PhysioNet 2018 (8 sur 12, dont un incertain), option exploratoire sur PhysioNet 2018 et comparaison
  équitable sur MESA ou SHHS après NSRR, règles communes fixées avant le calcul. À valider avec le tuteur.
- **Checkpoint préentraîné OSF vérifié** (`scripts/check_osf_checkpoint.py`, `results/osf_checkpoint_check.json`) :
  poids officiels (MIT, Hugging Face `f2067f2`) chargés en mode strict, 85,3 M paramètres, passage avant
  synthétique fini, empreinte et métadonnées enregistrées. Sorties identiques au code officiel avec ces
  vrais poids (écart 0, `results/osf_real_weights_equivalence.json`). Aucune évaluation sur patients.
- **Performances de la prédiction sur PSG brute mesurées** (`results/raw_grid_physionet2018.json`) :
  100 nuits de test, fenêtres d'un seul stade (83 %) : BiGRU κ 0,711 [0,678 ; 0,742], baseline 0,634.
  Pas de baisse par rapport à la grille annotée ; transitions non notées, score possiblement optimiste.
- **Évaluation de la prédiction sur PSG brute** : `scripts/embed_raw_grid.py` (embeddings des 100 nuits
  de test sur la grille fixe de 30 s, préchargement du sujet suivant, reprise, rejets motivés),
  `src/data/raw_grid_labels.py` (stade de l'expert par fenêtre, seulement pour noter ; fenêtres à cheval
  ou non annotées comptées à part), `scripts/evaluate_raw_grid.py` (modèles figés, κ et F1 avec IC par
  sujet, comparaison descriptive à la grille annotée) et notebook Kaggle
  `sleepfm_raw_grid_physionet2018.ipynb`. Aucun entraînement.
- **Bilan final vérifié** : `run_status.json` relit chaque cache demandé, contrôle
  les raisons des rejets et distingue les temporaires ; les seuls noms de fichiers
  ne suffisent plus pour annoncer un run complet. CLI et notebook refusent un bilan
  invalide avant succès, entraînement ou export ; les fichiers source restent intacts.
- **Contrôle des entrées OSF** (`scripts/check_osf_input.py`) : vérification du
  format d'un NPZ d'époques déjà préparées, ordre des canaux explicite, fréquence
  et durée strictes, refus des canaux dupliqués ou absents et des valeurs non finies.
  Rapport de provenance sans chargement de modèle ; aucun prétraitement clinique déduit.
- **Compatibilité du backbone OSF** : noms de projection d'attention alignés avec
  le code officiel, mode de découpage par groupes de canaux pris en charge,
  métadonnées et entrées invalides refusées. Sorties identiques à une référence
  officielle figée sur des tenseurs synthétiques ; aucun checkpoint préentraîné
  téléchargé ni évaluation sur patients.
- **Comparaison des caches** : les temps d'origine sont désormais comparés avant
  d'annoncer des fenêtres identiques ; temps absents signalés comme non vérifiés,
  tolérances absolues contrôlées et fichiers temporaires exclus du bilan.
- **Tableau consolidé lisible** : décimales à la française dans le Markdown (le CSV garde le point),
  champ manquant noté « absent », et rappel que la confirmation utilise les modèles figés du run à
  500 sujets (400 nuits, graine 42), que son JSON n'enregistre pas.
- **Consolidation des résultats sauvegardés** (`scripts/consolidate_results.py`) :
  tableaux CSV et Markdown des runs et confirmations terminées, sources SHA-256,
  effectifs et champs manquants explicites ; aucun modèle chargé ni score recalculé.
- **Validation avant travail** : listes de sujets non vides, uniques et sans chemins ;
  paramètres de lots invalides refusés avant calcul. Un cache existant sans manifeste
  ne reçoit plus automatiquement une provenance ; contrats incomplets refusés à l'import.
- **Reprise des lots** : contrat contrôlé avant toute suppression de cache corrompu,
  marqueur de rejet ou fichier temporaire. Nettoyage et `--retry-skipped` limités aux
  sujets demandés ; tailles et nombres de workers invalides refusés avant écriture.
- **PSG brute sans annotations** (`scripts/predict_raw_night.py`) : fichiers PhysioNet
  2018 `.hea`/`.mat`, grille fixe de 30 s depuis zéro, canaux et signal contrôlés,
  même contrat de signal et modèles exportés. CSV par nuit et JSON de provenance ;
  aucune modification des caches annotés. Les performances sur cette grille restent
  à évaluer. Écriture CSV atomique commune aux deux commandes de prédiction.
- **Confirmation sur les 494 sujets réservés** (passage unique, 2026-11-09) : écart de κ BiGRU moins
  baseline +0,094 [+0,086 ; +0,103], gain confirmé. Détail dans `docs/PROTOCOLE_CONFIRMATION.md` et
  `results/confirmation_physionet2018.json`.
- **Préchargement des données** : en mode par lots, le lot suivant se télécharge pendant l'extraction et
  les embeddings du lot en cours (au plus deux lots bruts sur disque). Résultats identiques ; une erreur
  de téléchargement arrête le run comme avant, sans laisser de dossier temporaire. Le temps affiché
  distingue la durée du téléchargement et l'attente réelle.
- **Durée de chaque étape** : `run_baseline.py` affiche, par lot, le temps de téléchargement, d'extraction
  et d'embeddings. Affichage seul, sans effet sur les calculs ; sert à repérer ce qui ralentit un run.
- **Protocole de confirmation validé** par le tuteur le 2026-11-06, critère compris (contenu inchangé).
- **Licence** : tous droits réservés (`LICENSE`), textes MIT de SleepFM et d'OSF reproduits dans `THIRD_PARTY_NOTICES.md`. Ne s'applique pas rétroactivement à ce qui a déjà été copié.
- **Commande de confirmation** (`scripts/confirm_frozen_models.py`), avec la liste figée des 494 sujets
  et les empreintes du protocole (`configs/`) : aucun entraînement, tout écart arrête le run avant le
  moindre score, un seul passage. Testée seulement sur des sujets déjà ouverts.
- **`--subjects-file`** : embeddings de sujets précis, sans sonde ni extension du découpage.
- **Bilan machine** (`run_status.json`) : ensemble exact demandé, réussi et rejeté, contrôlé par le notebook.
- **Variance des blocs d'embeddings** (`scripts/embedding_block_variance.py`) : calcul versionné.
- **Ablation par capteurs** (`scripts/ablation_modalites.py`) : baseline et BiGRU
  réentraînés sur chaque partie des embeddings (respiration, EEG/EOG, ECG), sans
  les recalculer. Exploratoire, une graine, sur des nuits de test déjà examinées :
  κ du BiGRU 0,664 avec tout, 0,669 avec l'EEG/EOG seul, 0,449 sans EEG/EOG. Ne
  simule ni un smartphone ni un objet connecté.
- **Graines et volume à 500 sujets** (`--train-nights`,
  `results/synthese_graines_volume_physionet2018_500.txt`) : sur 3 graines, κ du
  BiGRU complet de 0,664 à 0,682 ; gain sans ECG non établi ; perte sans EEG/EOG
  confirmée. Passer de 160 à 400 nuits : baseline +0,026 [+0,015 ; +0,037], BiGRU
  +0,012 à +0,035 selon la graine. Mêmes nuits de test, déjà examinées.
- **Protocole de confirmation** (`docs/PROTOCOLE_CONFIRMATION.md`), écrit avant
  d'ouvrir les 494 sujets réservés : modèles figés par empreinte, critère
  principal, exclusions et historique d'utilisation du test. À valider.
- **Import sûr des caches** (`src/data/cache_contract.py`) : le notebook compare
  le contrat complet de chaque source, signale tout conflit de fichier et garde
  la provenance de chaque fichier importé.
- **`--num-workers`** : workers du DataLoader séparés de `--num-threads`, 0 possible.
- **Traçabilité** : paramètres effectifs, nuits de validation et meilleur passage
  dans l'export (format 2) ; commit, état du code, versions et empreinte du
  découpage dans le JSON des résultats ; contrat aussi dans l'export du mode classique.
- **Prédiction sans stades** : `predict_night.py` accepte un cache sans `y`.
- **Écart apparié entre deux runs** dans `compare_test_nights.py`, seulement si
  les époques et les stades sont identiques.
- **Spectre de l'EEG par groupe d'époques** (`scripts/eeg_spectra_by_group.py`) :
  bandes de C3, alpha occipital et tonus du menton, par couple stade de
  l'expert / stade prédit, pour comprendre un désaccord.
- **Comparaison anciennes / nouvelles nuits de test**
  (`scripts/compare_test_nights.py`) : quand le découpage est étendu, κ sur
  les anciennes nuits de test, les nouvelles et l'ensemble, et nuits au κ le
  plus bas.
- **Découpage figé** (`--split-file`, `configs/split_physionet2018.json`) :
  un sujet garde son rôle quand on en ajoute ; les 80 / 20 sujets du run à
  100 sujets y sont enregistrés, étendus à 160 / 40 pour 200 sujets.
- **Manifeste du cache d'embeddings** (`manifest.json`) : empreinte du
  checkpoint, version du prétraitement, canaux, fréquence. Un cache d'un
  autre checkpoint est refusé, un autre prétraitement n'est jamais mélangé.
  Enregistré dans le JSON des résultats.
- **Temps d'origine de chaque époque** (`onset_s`) dans l'extraction et le
  cache ; **raison des rejets** enregistrée.
- **BiGRU pondéré** : option `class_weight="balanced"` de la sonde
  séquentielle et `scripts/compare_weighted_sequence.py`.
- **Notebook renommé** `notebooks/sleepfm_embeddings_physionet2018.ipynb`
  (« SleepFM — embeddings PhysioNet 2018 (GPU) ») ; règle de nommage des
  notebooks dans le GUIDE.
- **Export des sondes** (`--save-models`, `src/models/probe_export.py`) :
  sondes rechargeables avec leur manifeste (contrat d'entrée, classes,
  hyperparamètres, commit, versions) ; `scripts/predict_night.py` prédit
  l'hypnogramme d'une nuit en CSV, après contrôle du checkpoint et du
  prétraitement.
- **Comparaison de deux caches** (`scripts/compare_embedding_caches.py`) :
  mêmes époques, mêmes stades, embeddings identiques sur les sujets communs
  (pour vérifier le prétraitement version 2 sur les 100 premiers sujets).
- **CI** : tests aussi en Python 3.10, version minimale annoncée.
- **Examen des signaux bruts** (`scripts/inspect_night_signals.py`) : qualité
  des canaux et physiologie par stade (tonus du menton, mouvements oculaires,
  alpha). Deux nuits ratées : REM pris pour de l'éveil, EMG du menton quasi
  plat dans l'une, REM atypique dans l'autre.
- **`--embed-only`** (mode par lots) : calcule seulement les embeddings. Le
  notebook l'active par défaut : le GPU ne sert qu'à SleepFM, les sondes
  tournent ensuite sur le Mac (~20 à 25 min de GPU économisées par run).
- **Notebook** : 500 sujets par défaut (100 nuits de test, comme
  l'article) ; un Dataset Kaggle qui garde le zip des embeddings tel quel est
  décompressé avant l'import ; le découpage étendu est copié dans `Output`.
- **Notebook** (2026-11-04) : 200 sujets par défaut, dans `physionet2018_v2` avec le
  découpage figé ; sur Kaggle, n'importe que des embeddings au même
  prétraitement, et place prédictions et zip des embeddings dans `Output`.
- **Prédictions du test sauvegardées** (`<json>.predictions.npz`) : sujet,
  numéro d'époque, vrai stade, prédiction et probabilités de chaque sonde.
- **Ablation du contexte du BiGRU** (`scripts/ablation_sequence.py`) :
  contexte à la prédiction (nuit, 1 h, 5 min, 1 époque) et ordre mélangé.
- **Analyse des erreurs** (`scripts/analyse_errors.py`) à partir des
  prédictions sauvegardées : confusions, kappa par nuit, précision selon la
  distance au changement de stade, détail de N1.
- **AUPRC** (macro et par stade) et ROC-AUC par stade, dans la sortie et le
  JSON. La sortie compare la baseline officielle au tableau 8 de l'article
  SleepFM 2024 (PhysioNet 2018), stade par stade.

### Corrigé
- **Page du dépôt** : README raccourci avec résumé en anglais, résultats en bref et état du projet ; le détail passe, sans changement de chiffres, dans `docs/RESULTATS.md`. Chemins locaux retirés d'un fichier de résultats.
- **Import des caches** : destination avec des fichiers sans manifeste refusée, tous les conflits vérifiés
  avant copie, provenance avec SHA-256 écrite avant publication. Une interruption ne laisse que des fichiers
  vérifiés sous ce contrat.
- **Temps des caches récents** : au prétraitement version 2, `epoch_idx` et `onset_s` sont obligatoires, finis,
  positifs ou nuls et espacés d'au moins une époque (trous permis). Les caches historiques restent lisibles.
- **Téléchargement** : un transfert plus court que la taille annoncée est retenté et jamais gardé.
- **Comparaison entre runs** : temps d'origine comparés et doublons refusés ; sans temps, comparaison descriptive.
- **Ablation par capteurs** : les nouvelles archives gardent probabilités, classes, temps, sujets, paramètres,
  environnement et état du run, avec écriture atomique. Les archives du 6 nov. n'ont que les classes prédites :
  leurs AUROC ne se recalculent pas depuis ces fichiers.
- **Notebook** : bilan par identifiants, découpage sauvegardé hors du dépôt avant mise à jour et exporté dans
  les deux modes, révision du code explicite, `pip freeze` dans les sorties.
- **Poids aléatoires** : empreinte réelle du checkpoint synthétique, au lieu d'une étiquette commune.
- **Synthèse des graines** : « ne prédit jamais REM » valait pour le BiGRU seulement ; la baseline en prédit
  15 961 sur ce même bloc.
- **Documents** : découpage du guide (400 / 100), historique du protocole, version 2 du protocole.
- **Conclusions trop fortes** : le run à 500 sujets est présenté comme une
  évaluation élargie avec réutilisation partielle du test, pas comme une
  reproduction stricte ni un protocole aligné. L'effet du volume n'est établi que
  pour la baseline entre 160 et 400 nuits. Les nuits atypiques séparent mesures et
  hypothèses (`results/nuits_atypiques_physionet2018_500.txt`).
- **Spectre** : la plage 2-4 Hz manquait, et le « 80 % » additionnait deux
  médianes. Le delta couvre 0,5-4 Hz et la puissance lente se calcule par époque.
- **Arrondis du README** : un seul arrondi depuis le JSON (F1 du BiGRU 0,731, ROC-AUC du MLP 0,916).
- **Contrat d'inférence obligatoire** : manifeste absent ou incomplet, checkpoint,
  prétraitement, canaux, fréquence, classes ou dimension différents sont refusés
  avant de prédire, sauf `--allow-unverified-provenance`.
- **Import Kaggle** : deux sources de checkpoints différents pouvaient se mélanger
  sous un seul manifeste. Aucun mélange dans les résultats actuels, issus d'une seule source.
- **Validation des caches et du découpage** : temps partiellement manquants ou
  infinis, indices non entiers ou négatifs, labels à deux dimensions et partitions
  qui ne sont pas des listes sont refusés. Le BiGRU refuse aussi un temps manquant isolé.
- **Téléchargement** : fichier `.part` renommé à la fin et délai de réponse ; un
  transfert interrompu n'est plus pris pour un fichier complet.
- **Notebook** : le vrai code de retour du script est vérifié, et un run incomplet
  ou un JSON ancien ne produisent plus de message de succès.
- **Export des sondes** (`--save-models`) : le MLP, qui a aussi un attribut
  `model_`, était pris pour le BiGRU et l'export plantait. Seul un module
  torch est désormais déplacé sur CPU ; le test d'export couvre le MLP.
- **XGBoost sur Mac** : il plantait (segmentation fault) ou se bloquait
  dès que torch avait calculé dans le même processus (deux OpenMP
  différents, même sur un seul thread). `run_baseline.py` le retire des
  sondes sur Mac avec un avertissement ; ses tests sont sautés sur Mac et
  tournent en CI Linux.
- **Chargement OSF strict** : poids lus avec `weights_only=True`, et une clé
  manquante ou inattendue arrête le chargement (`allow_key_mismatch=True`
  pour un essai explicite).
- **Découpage figé validé** : sujet en double ou à la fois en train et en
  test refusé ; écriture atomique ; les nouveaux sujets ramènent la part du
  test à la fraction voulue (ajoutés un par un, ils allaient tous au train).
- **Caches vérifiés** : dimension, valeurs finies, stades connus, numéros
  d'époque uniques et temps croissants ; un cache invalide arrête le run.
- **Fin d'extraction** : fichiers de stades et de temps écrits de façon
  atomique et relus avant de sauter un sujet ; `--retry-skipped` retente les
  sujets rejetés.
- **Trous d'enregistrement** : le BiGRU et l'analyse d'erreurs coupent la
  nuit quand deux époques consécutives sont séparées de plus de 31 s (temps
  d'origine `onset_s`) ; sans ces temps (caches actuels), la nuit est
  supposée continue, avec un avertissement.
- **Scripts d'analyse** (ablation, BiGRU pondéré) : même chargement vérifié
  que le run principal (manifeste, découpage figé, caches).
- **Notebook** : s'arrête en erreur si le run n'a rien produit (le `| tee`
  masquait le code d'erreur).
- **Doc** : un seul run de référence dans le README ; formulations
  nuancées (checkpoint public non certifié identique à celui de l'article,
  « MLP essayé » plutôt que « frontière non linéaire »).
- **Début utile de l'enregistrement** (prétraitement version 2) : fin de la
  période « undefined » initiale, colonne désignée par son nom, comme le code
  officiel, au lieu du premier changement de la dernière colonne (« wake »).
  Identique sur les enregistrements vérifiés ; les caches actuels restent en
  version 1.
- **Extraction interrompue** : un dossier d'époques partiel n'est plus pris
  pour un sujet terminé ; le sujet est réextrait.
- **BiGRU** : le rembourrage des fenêtres courtes passait dans le GRU et
  influençait les vraies époques (sens retour). Le GRU reçoit maintenant la
  longueur réelle de chaque fenêtre. Effet mesuré négligeable (κ 0,6845 →
  0,6853).
- **Référence à l'article** : la documentation affirmait que SleepFM ne
  publiait pas de résultat sur PhysioNet 2018 et que la comparaison
  attendait MESA. L'article SleepFM 2024 évalue pourtant PhysioNet 2018
  (section 4.6, tableau 8 : ROC-AUC 0,924, AUPRC 0,759, F1 0,700). MESA
  devient une évaluation supplémentaire, pas un préalable.
- La formule « le test n'a servi qu'une fois » est précisée : le test ne sert
  à aucun choix, mais les mêmes nuits ont été évaluées par plusieurs runs.
- **Baseline** : le protocole officiel (`sleepfm/utils.py::train_model`)
  entraîne `LogisticRegression(class_weight="balanced")`. La baseline
  `logistic` est désormais pondérée ; l'ancienne, non pondérée, devient
  `logistic_unweighted`. Les résultats de la 0.1.0 venaient de la version
  non pondérée.
- **Convergence** : `max_iter` passe à 1 000 et la convergence de la
  régression logistique est vérifiée et affichée ; le notebook ne masque
  plus les avertissements.
- **Arrêt précoce du MLP** sur des nuits mises à part, au lieu d'époques
  tirées au hasard dans les mêmes nuits que l'entraînement (fuite).
- Chargement du checkpoint avec `weights_only=True`.
- **Reprise du mode par lots** : un `.npz` interrompu en pleine écriture
  restait sur le disque (ou le Drive), était pris pour un sujet traité et
  faisait planter le run suivant. L'écriture passe par un fichier
  temporaire renommé ensuite ; un cache illisible existant est détecté,
  supprimé et recalculé au démarrage.

### Résultats
- **500 sujets, 400 / 100 nuits** (2026-11-05), évaluation élargie, même nombre de
  nuits de test que l'article mais 40 déjà examinées : baseline pondérée κ 0,599 [0,570 ; 0,627], non pondérée
  0,613, MLP 0,591, BiGRU 0,664 [0,634 ; 0,694] (+0,065 [+0,047 ; +0,083],
  81 nuits sur 100). Article : ROC-AUC 0,921 / AUPRC 0,750 / F1 0,686 contre
  0,924 / 0,759 / 0,700. Sur les mêmes 36 163 époques du run à 200 sujets, écart
  apparié de 160 à 400 nuits, pour une graine par run : baseline +0,029
  [+0,008 ; +0,049], BiGRU +0,016 [−0,001 ; +0,031].
- 200 sujets, découpage figé étendu à 160 / 40 (2026-11-05), embeddings en
  prétraitement v2 calculés sur Kaggle : baseline pondérée κ 0,587
  [0,533 ; 0,643], non pondérée 0,608, MLP 0,586, BiGRU 0,656
  [0,594 ; 0,713] (+0,069 [+0,045 ; +0,096] sur la baseline, 33 nuits sur
  40). Sur les 20 anciennes nuits de test, les scores sont plus hauts qu'à 100
  sujets (baseline 0,620 contre 0,607, BiGRU 0,706 contre 0,685), avec un
  prétraitement également modifié (125 époques de plus) : comparaison
  descriptive. Deux nouvelles nuits de fort désaccord (κ 0,04 et 0,12) pèsent
  sur le total ; elles restent dans le test. Article : ROC-AUC 0,916 /
  AUPRC 0,750 / F1 0,679 contre 0,924 / 0,759 / 0,700.
- Deux nuits de fort désaccord examinées : tr03-1292 et tr04-0133. Mesures et
  hypothèses dans `results/nuits_atypiques_physionet2018_500.txt` ; ni erreur
  d'annotation ni erreur de modèle ne sont établies.
- Embeddings v2 contre v1 sur les 100 premières nuits : identiques sur 91 ;
  sur les 9 autres, la v2 récupère 480 époques de début de nuit, mêmes
  stades sur les époques communes.
- Comparaison à l'article SleepFM 2024 (tableau 8, PhysioNet 2018) :
  baseline officielle ROC-AUC 0,925 / AUPRC 0,770 / F1 0,683, contre
  0,924 / 0,759 / 0,700 publiés. Le profil du F1 par stade de la logistique
  non pondérée est le plus proche de l'article.
- Ablation du BiGRU sur 3 graines : gain de κ sur la baseline de +0,060 à
  +0,077 avec la nuit entière (toujours significatif), +0,03 à +0,04 avec
  l'ordre mélangé. L'ordre chronologique ajoute +0,02 à +0,04 en comparaison
  directe, significatif sur une graine sur trois : sa part exacte n'est pas
  établie (l'estimation « la moitié du gain » d'une seule graine est retirée).
  F1 de N1 du BiGRU : 0,46 à 0,48 selon la graine, pas de gain établi.
- Erreurs du BiGRU : gain surtout sur N2 et au milieu des segments stables,
  presque nul aux changements de stade ; rappel de N1 en baisse (0,47 contre
  0,53), plus souvent lissé en éveil.
- BiGRU pondéré : rappel de N1 0,58 (contre 0,47), F1 de N1 inchangé, κ
  0,629, soit −0,056 [−0,083 ; −0,033] face au BiGRU non pondéré, qui reste
  la référence.
- 100 sujets, split 80/20 par sujet, protocole officiel (2026-11-04) :
  baseline pondérée κ 0,606 [0,545 ; 0,669], non pondérée 0,646, MLP 0,623,
  BiGRU 0,684 [0,615 ; 0,739]. XGBoost éliminé par la sélection (gap F1
  +0,32). Mêmes chiffres sur GPU (Kaggle) et CPU (Colab), à 0,001 près.

### Ajouté
- Comparaison appariée de chaque sonde à la baseline, sur les mêmes nuits de
  test (bootstrap sur les sujets) : écart de κ et de F1 macro avec son IC,
  dans la sortie et le JSON (`vs_baseline`). Gain du BiGRU significatif
  (κ +0,078 [+0,041 ; +0,117]), pas celui du MLP.
- Intervalles de confiance à 95 % du kappa et du F1 macro par bootstrap sur
  les sujets de test.
- Matrice de confusion de la baseline (sortie et JSON).
- Notebook utilisable aussi sur Kaggle (détection automatique, import des
  embeddings depuis un Dataset Kaggle, résultats dans le panneau Output) ;
  il force aussi la mise à jour du code à la relance.
- Le JSON de résultats enregistre les sujets de train et de test, les
  réglages (sondes, `max_iter`, plis, seuils de sélection) et la convergence
  de chaque sonde : deux runs se comparent sans relire les logs.

## [0.1.0] — 2026-11-03

Premier jalon : baseline SleepFM reproduite de bout en bout sur PhysioNet 2018.

### Ajouté
- `run_baseline.py`, point d'entrée unique : split par sujet, extraction des
  époques, embeddings SleepFM figés (checkpoint officiel `best.pt`), sondes,
  évaluation, export JSON des métriques (`--results-json`).
- Mode par lots (`--physionet-subjects`, `--embeddings-dir`) : téléchargement
  au fil de l'eau, cache d'un `.npz` d'embeddings par sujet, reprise après
  interruption.
- Sondes comparées sur les mêmes embeddings : régression logistique
  (baseline), logistique pondérée, MLP, XGBoost, BiGRU sur la nuit.
- Sélection MLP / XGBoost par validation croisée par sujet sur le train,
  avec élimination en cas de surapprentissage ; test évalué une seule fois.
- Notebook Google Colab (GPU gratuit) avec cache sur Google Drive.
- Résultats : κ = 0,622, accuracy 0,722, F1 macro 0,681 sur 100 sujets
  (75 train / 25 test).
- CI (lint ruff puis tests), `pyproject.toml`, `pre-commit`, carte du modèle,
  guide d'utilisation.

### Corrigé
- Chargement du `best.pt` publié, dont les noms de clés diffèrent de ceux
  attendus par le script officiel.
- Commandes de démarrage du README (dossier de sortie du téléchargement).

### Retiré
- Random Forest (plus lent que XGBoost, moins adapté aux embeddings denses).
- `scripts/run_embedding_and_probe.py` et `src/eval/evaluate.py`, doublons de
  `run_baseline.py` et `src/models/linear_probe.py`.
- Dépendances inutilisées (`wfdb`, `matplotlib`, `pyyaml`, `pandas`).
