# Attributions

`src/models/effnet.py` and parts of `src/data/extract_physionet2018.py` and
`src/models/embeddings.py` are adapted from the official SleepFM codebase:

- Repository: https://github.com/rthapa84/sleepfm-codebase
- License: MIT (Copyright 2024 the SleepFM authors)

Adapted for readability and to remove a `sys.path` dependency on a
third_party/ clone; the model architecture, channel conventions, and
checkpoint key names are kept identical so official pretrained weights load
without remapping.

## OSF (src/models/osf_vit1d.py)

- Repository: https://github.com/yang-ai-lab/OSF-Open-Sleep-FM
- License: MIT

Minimal re-implementation of `osf/backbone/vit1d_cls.py::ViT` (1D ViT with
CLS token), enough to load the released `osf_backbone.pth` state_dict and
generate embeddings, without vendoring the full training/SSL codebase. See
`docs/OSF_COMPARISON.md` for the channel-convention and label-scheme
differences that must be handled before comparing SleepFM and OSF numbers
directly.
