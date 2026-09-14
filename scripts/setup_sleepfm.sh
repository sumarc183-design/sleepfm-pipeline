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
  git clone https://github.com/bdsp-core/OSF.git osf
fi

echo "Poids préentraînés à récupérer manuellement selon les instructions des dépôts respectifs"
echo "(voir third_party/sleepfm/README.md et third_party/osf/README.md)."
