# Accès aux données NSRR (MESA, SHHS, CHAT)

## Ce qu'est le NSRR

Le National Sleep Research Resource (sleepdata.org) est le dépôt qui héberge
MESA, SHHS et CHAT sous forme de données dé-identifiées pour la recherche
secondaire. L'accès n'est pas un simple téléchargement : il passe par un
accord d'utilisation des données (Data Use Agreement).

## Procédure (à revérifier sur sleepdata.org au moment de la demande — le
processus peut évoluer)

1. Créer un compte gratuit sur https://sleepdata.org
2. Sur la page du dataset visé (ex. `/datasets/mesa`), cliquer sur
   "Request access" / "Data Access"
3. Remplir la demande : objectif de recherche, affiliation, usage prévu
   (recherche non commerciale vs. usage commercial — à déclarer honnêtement,
   MESA a un comité de revue plus strict que d'autres cohortes NSRR)
4. Signer le Data Use Agreement : pas de tentative de ré-identification,
   pas de redistribution, citation du dataset dans toute publication/rapport
5. Délai d'approbation : généralement quelques jours pour un DUA standard ;
   MESA peut demander une revue supplémentaire (plusieurs semaines) car
   c'est une cohorte multi-centres avec son propre comité de publication
6. Une fois approuvé : téléchargement via le site NSRR ou script
   (rsync/AWS S3 selon le dataset)

## Priorité de dépôt

1. **MESA** — cette semaine (chemin critique pour reproduire la démo
   officielle SleepFM)
2. **SHHS** — en parallèle ou juste après (évaluation à plus grande échelle)
3. **CHAT** — phase 2 (pédiatrie)

## Action immédiate

Créer le compte sleepdata.org et déposer la demande MESA ne dépend
d'aucune autre étape technique — c'est purement administratif et c'est le
plus long délai de la phase 1. À lancer en parallèle du travail sur
PhysioNet 2018.
