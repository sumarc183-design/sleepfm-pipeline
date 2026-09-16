"""Télécharge un sous-ensemble du PhysioNet/CinC Challenge 2018 (PSG + labels de stades/arousals).

Nécessite le paquet `wfdb`. Le jeu complet fait ~130 Go ; utiliser --limit
pour ne récupérer qu'un sous-ensemble de sujets pendant la mise au point du pipeline.

Sur un réseau d'entreprise : si ce script reste bloqué sur "Downloading
files..." sans jamais avancer (alors qu'un navigateur classique accède
bien à physionet.org), c'est probablement un proxy géré par Windows/le
navigateur mais pas par la librairie Python. Solution de repli : chaque
enregistrement ne nécessite que 3 fichiers, téléchargeables directement au
navigateur depuis
https://physionet.org/files/challenge-2018/1.0.0/training/<id>/<id>.hea
(+ .mat + -arousal.mat), à ranger dans
data/raw/physionet2018/training/<id>/.
"""
import argparse
import os

import wfdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Répertoire de sortie")
    parser.add_argument("--limit", type=int, default=10, help="Nombre de sujets à télécharger")
    parser.add_argument(
        "--split",
        default="training",
        choices=["training", "test"],
        help="Sous-ensemble PhysioNet 2018",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    records = wfdb.get_record_list("challenge-2018/1.0.0")
    subset = [r for r in records if r.startswith(args.split)][: args.limit]

    print(f"{len(subset)} enregistrement(s) à télécharger vers {args.out}")
    for record in subset:
        wfdb.dl_database(
            "challenge-2018",
            dl_dir=args.out,
            records=[record],
            keep_subdirs=True,
        )


if __name__ == "__main__":
    main()
