# Comparaison respiratoire OSF / SleepFM — plan du 12 novembre 2026

Plan écrit avant l'ajustement et la lecture des scores OSF pour cette tâche.
La baseline SleepFM est déjà connue. Il s'agit d'une exploration sur des
sujets examinés auparavant, pas d'une confirmation indépendante.

## Question et critère principal

La même logistique entraînée sur les embeddings OSF classe-t-elle mieux les
fenêtres avec apnées/hypopnées que la baseline SleepFM déjà sauvegardée ?

**Critère principal : AP OSF moins AP SleepFM sur les mêmes fenêtres de test.**
IC à 95 % par bootstrap **apparié par sujet**, 1 000 tirages, graine 42.
OSF au-dessus si la borne basse est > 0 ; SleepFM au-dessus si la borne haute
est < 0 ; sinon aucune différence établie. Publier quelle que soit l'issue.

## Règles inchangées

- Réutiliser les labels et l'inventaire de la première baseline respiratoire.
  Aucun nouveau téléchargement ni modification des annotations.
- Cible : union des apnées obstructives, centrales, mixtes et hypopnées.
  Positif si au moins 10 s d'événement, négatif si 0 s ; sinon exclu.
- Mêmes **360 sujets d'ajustement, 40 de validation, 100 de test** ; aucun
  usage des 494 sujets de confirmation de stadification ni de NSRR.
- Vérifier les empreintes et aligner par sujet/indice ; refuser doublons,
  fenêtres manquantes, stades différents et débuts décalés de plus de 0,01 s.
- OSF figé, embeddings CLS existants (768 dimensions). Logistique C = 1,
  pondération équilibrée, maximum 1 000 itérations, aucun balayage.
- Seuil F1 choisi sur la validation seulement, plus grand seuil en cas
  d'égalité ; aucun réajustement après le choix du seuil.
- Conserver la baseline SleepFM publiée (1 536 dimensions), ses probabilités
  et son seuil, sans réentraînement ni sélection d'une meilleure variante.
- Secondaires : AP et ROC-AUC, F1, précision, sensibilité, spécificité,
  matrices de confusion et durée d'ajustement. Les seuils de chaque modèle
  n'imposent pas la même sensibilité ; le critère principal est sans seuil.

## Limites et sorties

La comparaison porte sur les pipelines existants : dimensions, échelles et
prétraitements diffèrent. Quatre des douze canaux OSF sont mis à zéro ;
correspondance AIRFLOW/NP et normalisation par nuit restent des hypothèses.
Ce résultat ne sépare pas l'effet de l'architecture, du pré-entraînement et
du prétraitement. Une supériorité en stadification ne préjuge pas du résultat
respiratoire. Les scores ne sont ni un IAH ni un diagnostic d'apnée obstructive.

Sauvegarder JSON, probabilités alignées, modèle OSF et seuil, inventaire des
entrées et sources exactes exécutées. Ne pas écraser la baseline ni les
résultats antérieurs. Les tirages bootstrap sans les deux classes sont
comptés et exclus des IC ; échec si aucun tirage ne peut être évalué.

Ce fichier restera inchangé après le calcul. Résultat et commandes dans
`docs/RESPIRATORY_RESULTATS.md` et le guide.
