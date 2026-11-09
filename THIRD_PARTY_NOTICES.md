# Attributions

## SleepFM (`src/models/effnet.py`, parties de `src/data/extract_physionet2018.py` et de `src/models/embeddings.py`)

- Dépôt : https://github.com/rthapa84/sleepfm-codebase
- Licence : MIT (Copyright 2024 les auteurs de SleepFM)

Adapté pour la lisibilité et pour supprimer la dépendance à un clone ajouté
à `sys.path` dans third_party/. L'architecture du modèle, les conventions de
canaux et les noms de clés du checkpoint sont strictement identiques, pour
que les poids préentraînés officiels se chargent sans renommage.

## OSF (`src/models/osf_vit1d.py`)

- Dépôt : https://github.com/yang-ai-lab/OSF-Open-Sleep-FM
- Licence : MIT

Adaptation de `osf/backbone/vit1d_cls.py::ViT` (signal PSG, token CLS,
découpage temporel ou par groupes de canaux). Noms des poids et sorties
vérifiés contre la source au commit `d75d616736a52733d8565edba9e0fd07dab41c4a`
sur des tenseurs synthétiques ; checkpoint préentraîné non testé ici. Voir
`docs/OSF_COMPARISON.md` pour les différences de canaux et de schéma de
labels à traiter avant de comparer directement SleepFM et OSF.

## Textes des licences

Ces licences couvrent seulement les parties adaptées citées ci-dessus. Le reste du dépôt suit le fichier [LICENSE](LICENSE).

### SleepFM

```text
Copyright 2024 by the author(s)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### OSF

```text
MIT License

Copyright (c) 2026 Health Intelligence Lab @ UCLA (https://github.com/yang-ai-lab)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
