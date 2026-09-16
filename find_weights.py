#!/usr/bin/env python3
"""
SleepFM & RETFound Weights Discovery
=====================================
Find and verify official model weights from GitHub and HuggingFace
"""

print("=" * 70)
print("FOUNDATION MODEL WEIGHTS DISCOVERY")
print("=" * 70)

# ============================================================================
# 1. SleepFM
# ============================================================================
print("\n1️⃣  SLEEPFM (Sleep Staging Foundation Model)")
print("-" * 70)

sleepfm_sources = {
    "Paper": "https://doi.org/10.1038/s41591-025-04133-4",
    "GitHub Repo": "https://github.com/braindecode/braindecode",
    "GitHub Releases": "https://github.com/braindecode/braindecode/releases",
    "BrainDecode Docs": "https://braindecode.org/",
}

print("\n📚 OFFICIAL SOURCES:")
for name, url in sleepfm_sources.items():
    print(f"   {name:.<30} {url}")

print("\n🔍 VERIFY AVAILABILITY:")
print("   ✅ GitHub: Check releases for .pth files")
print("   ✅ BrainDecode PyPI: pip install braindecode")
print("   ❓ Model auto-download: Check model.py for HF model ids")

# ============================================================================
# 2. RETFound
# ============================================================================
print("\n\n2️⃣  RETFOUND (Retinal Foundation Model)")
print("-" * 70)

retfound_sources = {
    "GitHub Repo": "https://github.com/openmedlab/RETFound_MAE",
    "GitHub Releases": "https://github.com/openmedlab/RETFound_MAE/releases",
    "Paper": "https://doi.org/10.48550/arXiv.2310.08606",
}

print("\n📚 OFFICIAL SOURCES:")
for name, url in retfound_sources.items():
    print(f"   {name:.<30} {url}")

print("\n🔍 VERIFY AVAILABILITY:")
print("   ✅ GitHub: Check releases for .pth files (CFP & OCT models)")
print("   ✅ Model zoo: Check repo structure for checkpoint urls")
print("   ✅ Expected files: RETFound_cfp_weights.pth, RETFound_oct_weights.pth")

# ============================================================================
# 3. Actions
# ============================================================================
print("\n\n📋 IMMEDIATE ACTIONS")
print("-" * 70)
print("""
PHASE 1 (TODAY):
  □ Visit GitHub releases for both repos
  □ Download latest .pth files
  □ Verify SHA/checksums if provided
  □ Test load with torch.load()

PHASE 2 (DAY 1-2):
  □ Create baseline scripts for SleepFM on PhysioNet 2018
  □ Create baseline scripts for RETFound on RFMiD/ODIR
  □ Test inference on dummy data
  □ Measure model size & inference time

PHASE 3 (DAY 3-7):
  □ Download PhysioNet 2018 EEG dataset
  □ Preprocess & create DataLoaders
  □ Run baseline evaluation (AUROC, accuracy, etc.)
  □ Document results
""")

print("\n" + "=" * 70)
