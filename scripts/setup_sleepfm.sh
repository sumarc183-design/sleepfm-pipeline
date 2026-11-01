#!/usr/bin/env bash
# Clone le code officiel SleepFM (et optionnellement OSF) dans third_party/,
# pour réutiliser leurs scripts de prétraitement et charger les poids préentraînés.
set -euo pipefail

mkdir -p third_party
cd third_party

if [ ! -d sleepfm ]; then
  git clone https://github.com/rthapa84/sleepfm-codebase.git sleepfm
fi

if [ ! -d osf ]; then
  git clone https://github.com/yang-ai-lab/OSF-Open-Sleep-FM.git osf
fi

echo "Checkpoint SleepFM : third_party/sleepfm/sleepfm/checkpoint/best.pt (versionné dans le dépôt officiel)."
echo "Poids OSF à récupérer manuellement (voir third_party/osf/README.md)."
