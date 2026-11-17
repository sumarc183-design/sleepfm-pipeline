# Plan — examen des hypopnées avec le médecin (17 novembre 2026)

**À valider avant toute sélection. Examen qualitatif et descriptif : aucun modèle, aucun seuil ne change.**

## Pourquoi

Les hypopnées sont la principale limite d'OSF : 3 185 des 4 319 fenêtres positives manquées
(73,7 %), rappel 48,9 % contre 54,1 % pour SleepFM, IC de l'écart contenant zéro
([compte rendu du 12-11](../sleepfm-pipeline/docs/resultats/respiratoire_erreurs_2026-11-12.md)).
Le stade et le type d'événement n'expliquent pas ces erreurs. Le tuteur a validé le 12-11 de les
examiner avec un médecin. Question posée : **les hypopnées manquées sont-elles des événements
nets que le modèle rate, ou des annotations limites, peu visibles sur les signaux ?**

## Échantillon (tiré une seule fois, graine 42, avant tout affichage)

Source : les 100 nuits de test, fenêtres de 30 s, prédictions figées
(`results/respiratory_osf_vs_sleepfm_physionet2018.predictions.npz`) et groupes d'événements
(`results/respiratory_errors_physionet2018.groups.npz`). Seuil OSF « initial » figé : 0,7636.

| Groupe | Définition | Fenêtres |
|---|---|---:|
| A — hypopnées manquées | « hypopnée seule », probabilité OSF < seuil | 20 |
| B — hypopnées détectées (témoins) | « hypopnée seule », probabilité OSF ≥ seuil | 10 |
| C — fausses alertes | aucun événement annoté, probabilité OSF ≥ seuil | 10 |

- Au plus une fenêtre par nuit dans chaque groupe, pour qu'aucune nuit ne domine.
- Tirage aléatoire simple parmi les fenêtres éligibles, nuits tirées d'abord puis une fenêtre par nuit.
- **Total 40 fenêtres, environ 45 min de relecture** (environ 1 min par fenêtre).
- Les 5 nuits les plus difficiles du compte rendu (dont tr06-0193) ne sont pas ajoutées à l'échantillon :
  elles ont été choisies sur les erreurs. Elles pourront être montrées à part, comme illustration.

## Ce que voit le médecin

- 2 minutes centrées sur la fenêtre de 30 s (encadrée) : débit nasal, sangle thoracique,
  sangle abdominale, SaO2, une voie EEG (C4-M1, pour les micro-éveils).
- Les annotations des experts PhysioNet en surimpression.
- **Sans** la sortie des modèles ni le groupe : les 40 fenêtres sont mélangées dans un ordre
  aléatoire figé, pour ne pas orienter le jugement.

## Questions, pour chaque fenêtre

1. Un événement respiratoire est-il visible dans la fenêtre encadrée ? aucun / hypopnée / apnée / incertain
2. S'il y a une hypopnée annotée : les critères AASM semblent-ils remplis (baisse du débit ≥ 30 %
   pendant ≥ 10 s, avec désaturation ≥ 3 % ou micro-éveil) ? oui / non / incertain
3. Qualité du signal respiratoire : bonne / artefact / capteur décroché
4. Commentaire libre (facultatif)

## Format

Page HTML autonome, sans réseau, transmise au médecin. Les réponses sont saisies dans la page et
exportées en un fichier CSV qu'il renvoie. Données PhysioNet 2018 (accès libre) ; aucun signal de
la partie réservée ni de NSRR. Seuls les signaux bruts des nuits tirées (au plus 40) sont
retéléchargés.

## Lecture prévue (fixée avant)

Proportions par groupe, descriptives, sans test (effectifs trop faibles) :
- part des fenêtres A jugées « hypopnée nette, critères remplis » contre la même part en B ;
- part des fenêtres C où le médecin voit un événement non annoté ;
- part des fenêtres à signal de mauvaise qualité dans chaque groupe.

Ces résultats orientent la suite (cible, annotations, préparation des signaux). Ils ne servent à
régler aucun seuil sur le test : tout réglage se ferait sur train/validation avec une évaluation distincte.

## À valider par Marc

1. Taille et composition de l'échantillon (20 / 10 / 10).
2. Médecin à l'aveugle sur les modèles (oui proposé).
3. Questions et voies affichées.
4. Qui transmet la page au médecin et dans quel délai.
