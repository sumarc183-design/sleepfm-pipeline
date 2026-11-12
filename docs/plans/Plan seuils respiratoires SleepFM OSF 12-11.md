# SleepFM / OSF — plan des seuils respiratoires, 12 novembre 2026

Complément exploratoire sur les partitions respiratoires existantes : 360 nuits
d'ajustement, 40 de validation, 100 de test déjà examinées. Les modèles figés
et leurs seuils initiaux restent inchangés. Aucun entraînement ni nouvelle donnée.

## Règles fixées avant le calcul

- Recharger les deux régressions respiratoires déjà enregistrées, après
  vérification de leurs empreintes et contrats. Lire uniquement les embeddings
  des 40 nuits de validation ; réutiliser les probabilités de test archivées.
- Vérifier les partitions, inventaires, empreintes des 40 caches de validation,
  labels et alignement SleepFM/OSF. Exiger l'identité des fenêtres, probabilités
  SleepFM et groupes du test avec les références précédentes.
- Calculer les probabilités de validation avec les modèles rechargés. Refaire
  la sélection F1 initiale comme contrôle : tolérance absolue 0,000002 sur le
  seuil, fixée ici pour les écarts numériques déjà observés lors du rechargement.
  Enregistrer l'écart ; arrêter au-delà. Le seuil F1 conservé reste celui de
  l'archive, sans le remplacer par ce recalcul.
- Trois cibles de sensibilité : **70 %, 80 %, 90 %**, choisies pour illustrer
  le compromis, sans valeur de recommandation clinique. Pour chaque modèle,
  retenir le plus grand seuil qui atteint la cible sur la validation, avec
  décision `score >= seuil`. Les ex æquo sont inclus ensemble.
- Fixer tous ces seuils avant de calculer leurs métriques de test. Aucun
  ajustement pour atteindre la cible sur le test ; une cible peut y être manquée.
- Rapporter, pour le seuil F1 initial et les trois seuils : sensibilité,
  spécificité, précision, F1, matrice et nombre de fenêtres positives prédites.
  Compter séparément les fenêtres de fausse alerte, sans les appeler événements.
- Décrire aussi le rappel sur les fenêtres d'hypopnée seules, avec les groupes
  déjà figés de l'analyse des erreurs. Aucun réglage spécifique à ce sous-groupe.
- IC à 95 % des sensibilités, spécificités, précisions et rappels hypopnées :
  bootstrap des 100 nuits de test, 500 tirages, graine 42, seuils maintenus fixes.
  Ces IC ne couvrent pas l'incertitude du choix du seuil sur la validation.
- Retrouver les métriques de test des seuils initiaux avant de publier. Garder
  probabilités de validation, seuils, empreintes, code et présent plan.

## Limites

Analyse par fenêtres sur PSG annotée, pas détection d'événements ni IAH,
diagnostic ou validation d'une alerte clinique. La prévalence affecte la précision.
Les scores logistiques ne sont pas des risques cliniques calibrés. Les nuits de
test sont déjà explorées ; aucune confirmation indépendante ni supériorité
statistique entre seuils n'est revendiquée. OSF conserve ses quatre canaux absents
et sa normalisation supposée. Ni NSRR ni 494 sujets de confirmation utilisés.

Plan et compte rendu datés dans `Journal`, artefacts dans le projet. La référence
de code actuelle est `921758d` ; aucune publication Git n'est prévue par ce plan.
