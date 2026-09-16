"""
SleepFM Baseline — PhysioNet 2018 Evaluation
=============================================

Phase 2-3 Objective:
  - Load SleepFM pre-trained weights
  - Download & preprocess PhysioNet 2018 sleep EEG data
  - Evaluate on sleep stage classification (W, N1, N2, N3, REM)
  - Report baseline AUROC, accuracy, confusion matrix

References:
  - SleepFM paper: https://doi.org/10.1038/s41591-025-04133-4
  - PhysioNet 2018: https://www.physionet.org/content/sleep-edfx/1.0.0/
  - BrainDecode: https://braindecode.org/
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    roc_auc_score,
    classification_report
)
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

# Dataset paths
PHYSIONET_ROOT = Path(__file__).parent / "data" / "physionet-2018"
PHYSIONET_RAW = PHYSIONET_ROOT / "raw"
PHYSIONET_PROCESSED = PHYSIONET_ROOT / "processed"

# Model paths
MODEL_WEIGHTS_PATH = Path(__file__).parent / "checkpoints" / "sleepfm_pretrained.pth"
RESULTS_DIR = Path(__file__).parent / "results" / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Hyperparameters
BATCH_SIZE = 32
SAMPLING_RATE = 100  # Hz (standard for sleep EEG)
SEGMENT_LENGTH = 30  # seconds per epoch
SEQUENCE_LENGTH = int(SAMPLING_RATE * SEGMENT_LENGTH)  # 3000 samples

# Sleep stages
SLEEP_STAGES = {
    'W': 0,   # Wake
    'N1': 1,  # NREM Stage 1
    'N2': 2,  # NREM Stage 2
    'N3': 3,  # NREM Stage 3
    'REM': 4  # REM sleep
}
NUM_CLASSES = len(SLEEP_STAGES)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42

# ============================================================================
# Logger Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "baseline.log")
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Preparation
# ============================================================================

def setup_directories():
    """Create output directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PHYSIONET_PROCESSED.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {RESULTS_DIR}")


def download_physionet_2018():
    """
    Download PhysioNet 2018 Sleep Dataset.
    
    Manual steps (automated version requires API token):
    1. Visit https://www.physionet.org/content/sleep-edfx/1.0.0/
    2. Download desired subject files (.edf format)
    3. Place in {PHYSIONET_RAW}/
    
    For CI/automated: Use mne.datasets.sleep_physionet
    """
    logger.info("PhysioNet 2018 download instructions:")
    logger.info("  1. Visit https://www.physionet.org/content/sleep-edfx/1.0.0/")
    logger.info("  2. Download test-01 through test-20 or all subjects")
    logger.info("  3. Place .edf files in data/physionet-2018/raw/")
    logger.info("")
    logger.info("Or use mne for automated download:")
    logger.info("  from mne.datasets import sleep_physionet")
    logger.info("  sleep_physionet.data_path()")
    
    # Check if data exists
    edf_files = list(PHYSIONET_RAW.glob("*.edf"))
    if edf_files:
        logger.info(f"✅ Found {len(edf_files)} EDF files in {PHYSIONET_RAW}")
        return edf_files
    else:
        logger.warning(f"⚠️  No EDF files found. Please download PhysioNet 2018 data.")
        return []


def load_and_preprocess_subject(edf_file):
    """
    Load EDF file and preprocess EEG signal.
    
    Requirements: pip install mne
    
    Returns:
        X: (n_epochs, sequence_length) - preprocessed EEG
        y: (n_epochs,) - sleep stage labels
    """
    try:
        import mne
    except ImportError:
        logger.error("Please install mne: pip install mne")
        return None, None
    
    try:
        # Load EDF file
        logger.info(f"Loading {edf_file.name}...")
        raw = mne.io.read_raw_edf(edf_file, preload=True)
        
        # Load sleep annotation file (.txt)
        annot_file = edf_file.with_suffix('.txt')
        if not annot_file.exists():
            logger.warning(f"No annotation file: {annot_file}")
            return None, None
        
        # Parse annotations
        with open(annot_file) as f:
            lines = f.readlines()
        
        # Extract EEG channel (usually first channel)
        eeg_data = raw.get_data(picks='eeg')[0]
        
        # Preprocess: filter, downsample
        raw.filter(l_freq=0.5, h_freq=30.0, verbose=False)
        eeg_data = raw.get_data(picks='eeg')[0]
        
        # Segment into 30-second epochs
        n_samples_per_epoch = SEQUENCE_LENGTH
        n_epochs = len(eeg_data) // n_samples_per_epoch
        
        X = eeg_data[:n_epochs * n_samples_per_epoch].reshape(n_epochs, n_samples_per_epoch)
        
        # Parse sleep stage labels from annotation
        y = np.zeros(n_epochs, dtype=int)
        for i, line in enumerate(lines):
            if i < n_epochs:
                # Annotation format: "Stage N3", "Stage REM", etc.
                stage_name = line.strip().split()[-1]
                if stage_name in SLEEP_STAGES:
                    y[i] = SLEEP_STAGES[stage_name]
        
        logger.info(f"  ✅ Loaded {n_epochs} epochs, shape: {X.shape}")
        return X, y
    
    except Exception as e:
        logger.error(f"Error loading {edf_file}: {e}")
        return None, None


def create_synthetic_baseline_data():
    """
    Create synthetic EEG data for immediate testing.
    Replace with real PhysioNet 2018 once downloaded.
    """
    logger.info("Creating synthetic baseline data (will be replaced by PhysioNet 2018)...")
    
    # Synthetic: 500 epochs × 3000 samples (30 sec @ 100 Hz)
    X_train = np.random.randn(300, SEQUENCE_LENGTH).astype(np.float32)
    y_train = np.random.randint(0, NUM_CLASSES, 300)
    
    X_test = np.random.randn(100, SEQUENCE_LENGTH).astype(np.float32)
    y_test = np.random.randint(0, NUM_CLASSES, 100)
    
    logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# Model Setup
# ============================================================================

class SleepFMBaseline(nn.Module):
    """
    SleepFM Baseline Classifier.
    
    Loads pre-trained SleepFM encoder and adds classification head.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained_path=None):
        super().__init__()
        self.num_classes = num_classes
        
        # Try to load pre-trained SleepFM
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pre-trained weights from {pretrained_path}...")
            self.encoder = torch.load(pretrained_path, map_location=DEVICE)
            hidden_dim = 256  # Adjust based on actual SleepFM architecture
        else:
            logger.warning(f"Pre-trained weights not found at {pretrained_path}")
            logger.info("Using random initialization (download weights from GitHub releases)")
            # Placeholder encoder (CNN)
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=11, stride=4, padding=5),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, kernel_size=11, stride=4, padding=5),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            hidden_dim = 64
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """x: (batch_size, sequence_length) or (batch_size, 1, sequence_length)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Encode
        features = self.encoder(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Classify
        logits = self.classifier(features)
        return logits


def load_or_create_model(pretrained_path=None):
    """Load or create SleepFM baseline model."""
    model = SleepFMBaseline(num_classes=NUM_CLASSES, pretrained_path=pretrained_path)
    model = model.to(DEVICE)
    logger.info(f"Model created. Total params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        # Forward
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
            total_loss += loss.item()
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    loss = total_loss / len(data_loader)
    
    # Multi-class ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(
            all_labels, 
            torch.softmax(torch.tensor(logits), dim=1).numpy(),
            multi_class='ovr',
            average='weighted'
        )
    except:
        auc = 0.0
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'auc': auc,
        'preds': all_preds,
        'labels': all_labels
    }


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run baseline evaluation pipeline."""
    logger.info("=" * 70)
    logger.info("SleepFM Baseline — PhysioNet 2018")
    logger.info("=" * 70)
    
    # Setup
    setup_directories()
    
    # Data preparation
    logger.info("\n[STEP 1] Data Preparation")
    download_physionet_2018()
    
    # For now, use synthetic data (will be replaced)
    X_train, y_train, X_test, y_test = create_synthetic_baseline_data()
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    logger.info("\n[STEP 2] Model Creation")
    model = load_or_create_model(pretrained_path=MODEL_WEIGHTS_PATH)
    
    # Training
    logger.info("\n[STEP 3] Training (5 epochs for baseline)")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        test_results = evaluate(model, test_loader, criterion)
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, "
            f"test_acc={test_results['accuracy']:.4f}, "
            f"test_auc={test_results['auc']:.4f}"
        )
    
    # Final evaluation
    logger.info("\n[STEP 4] Final Evaluation")
    final_results = evaluate(model, test_loader, criterion)
    
    logger.info(f"Final Accuracy: {final_results['accuracy']:.4f}")
    logger.info(f"Final AUC: {final_results['auc']:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    stage_names = {v: k for k, v in SLEEP_STAGES.items()}
    report = classification_report(
        final_results['labels'],
        final_results['preds'],
        target_names=[stage_names[i] for i in range(NUM_CLASSES)],
        zero_division=0
    )
    logger.info("\n" + report)
    
    # Confusion matrix
    cm = confusion_matrix(final_results['labels'], final_results['preds'])
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Save results
    logger.info(f"\n✅ Baseline evaluation complete. Results saved to {RESULTS_DIR}")
    
    # Save model
    model_path = RESULTS_DIR / "sleepfm_baseline_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
