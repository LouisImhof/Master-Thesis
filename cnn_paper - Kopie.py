# cnn_paper.py - Jiang, Kelly & Xiu (2021) "(Re-)Imag(in)ing Price Trends"
# Exact 3-block VGG-style CNN on 60×64 grayscale OHLC images
# Run gen_baseline.py first to create images_baseline/{train,val,test}/
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import time
from sklearn.metrics import roc_auc_score, f1_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_DIR  = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_baseline\images'
IMG_H    = 64
IMG_W    = 60
IN_CH    = 1

LR       = 1e-4   # raised from 1e-5 — too slow to escape majority-class attractor
BATCH    = 512    # small model → large batch fully utilises GPU
PATIENCE = 5      # increased from 2 — too aggressive with only 50 stocks
N_RUNS   = 5

MODEL_DIR = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_baseline\models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── DATASET ───────────────────────────────────────────────────────────────────
class ChartDataset(Dataset):
    """Loads all images into RAM once — eliminates I/O bottleneck during training."""
    def __init__(self, folder):
        files  = sorted(f for f in os.listdir(folder) if f.endswith('.png'))
        labels = [int(f.split('_')[-1].replace('.png', '')) for f in files]
        pos = sum(labels); neg = len(labels) - pos
        print(f"  {os.path.basename(folder):5s}: {len(files):,} images  "
              f"UP={pos:,} ({pos/len(labels)*100:.1f}%)  "
              f"DOWN={neg:,} ({neg/len(labels)*100:.1f}%)"
              f"  → caching to RAM...", end='', flush=True)

        n = len(files)
        self.images = np.empty((n, IMG_H, IMG_W), dtype=np.float32)
        for i, fname in enumerate(files):
            self.images[i] = np.array(
                Image.open(os.path.join(folder, fname)), dtype=np.float32
            ) / 255.0
        self.labels = np.array(labels, dtype=np.int64)
        mb = self.images.nbytes / 1e6
        print(f" {mb:.0f} MB")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.images[idx]).unsqueeze(0),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ── MODEL ─────────────────────────────────────────────────────────────────────
class JiangCNN(nn.Module):
    """
    3-block architecture from Jiang, Kelly & Xiu (2021).

    Block 1: Conv(5×3, stride_h=3, dilation_h=2) → BN → LeakyReLU(0.01) → MaxPool(2×1)
    Block 2: Conv(5×3) → BN → LeakyReLU(0.01) → MaxPool(2×1)
    Block 3: Conv(5×3) → BN → LeakyReLU(0.01) → MaxPool(2×1)
    Channels: 1 → 64 → 128 → 256
    FC: Flatten → Dropout(0.5) → Linear(fc_in, 2)

    For 60×64 input: padding=(7,1) in block 1 gives fc_in = 46,080
    """
    def __init__(self, in_ch=1, img_h=64, img_w=60):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — stride=3 and dilation=2 in height only
            nn.Conv2d(in_ch, 64,  (5, 3), stride=(3, 1), dilation=(2, 1), padding=(7, 1)),
            nn.BatchNorm2d(64),  nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 1)),
            # Block 2
            nn.Conv2d(64,  128, (5, 3), padding=(2, 1)),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 1)),
            # Block 3
            nn.Conv2d(128, 256, (5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 1)),
        )
        with torch.no_grad():
            fc_in = self.features(torch.zeros(1, in_ch, img_h, img_w)).flatten(1).shape[1]
        print(f"  FC input dim: {fc_in:,}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(fc_in, 2),
        )
        self._xavier_init()

    def _xavier_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


# ── TRAIN ONE RUN ─────────────────────────────────────────────────────────────
def train_one_run(run_id, train_loader, val_loader, test_loader, device):
    model     = JiangCNN(IN_CH, IMG_H, IMG_W).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  # batches are already balanced via WeightedRandomSampler
    use_amp   = device.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_auc = 0.0
    patience_cnt = 0
    best_state   = None

    print(f"\n  Run {run_id+1}  AMP={'ON' if use_amp else 'OFF'}")

    for epoch in range(50):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(X), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(X)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(y.numpy())
        val_auc = roc_auc_score(np.array(val_labels), np.array(val_probs))

        print(f"    Epoch {epoch+1:2d}  Train={train_loss:.4f}  ValAUC={val_auc:.4f}  "
              f"({time.time()-t0:.0f}s)")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, f'{MODEL_DIR}/run_{run_id+1}.pth')
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stop epoch {epoch+1}  best ValAUC={best_val_auc:.4f}")
                break

    # ── Evaluate on val + OOS test set ────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    def evaluate(loader):
        p_list, l_list = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(X)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                p_list.extend(probs)
                l_list.extend(y.numpy())
        p = np.array(p_list); l = np.array(l_list)
        return p, l, {
            'auc':   roc_auc_score(l, p),
            'acc':   ((p > 0.5) == l).mean(),
            'f1':    f1_score(l, p > 0.5, average='macro', zero_division=0),
            'brier': np.mean((p - l) ** 2),
        }

    _, _, val_m  = evaluate(val_loader)
    test_preds, test_labels, test_m = evaluate(test_loader)

    print(f"  [VAL 2019-20]  Acc: {val_m['acc']*100:.2f}%"
          f"  AUC: {val_m['auc']:.4f}"
          f"  F1: {val_m['f1']:.4f}"
          f"  Brier: {val_m['brier']:.4f}")
    print(f"  [OOS 2021-26]  Acc: {test_m['acc']*100:.2f}%"
          f"  AUC: {test_m['auc']:.4f}"
          f"  F1: {test_m['f1']:.4f}"
          f"  Brier: {test_m['brier']:.4f}")
    return test_preds, test_m['auc'], test_labels


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print('  Jiang, Kelly & Xiu (2021)  Baseline CNN')
    print(f'  Image: {IMG_W}×{IMG_H}px grayscale  |  I20R20  |  3-block VGG')
    print(f'  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU  : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'  LR={LR}  Batch={BATCH}  Patience={PATIENCE}  Runs={N_RUNS}')
    print(f'  Split: Train<2019 | Val 2019-21 | OOS>=2021')
    print('=' * 60)

    print('\nLoading datasets...')
    train_ds = ChartDataset(f'{IMG_DIR}/train')
    val_ds   = ChartDataset(f'{IMG_DIR}/val')
    test_ds  = ChartDataset(f'{IMG_DIR}/test')

    # Balanced sampler: each class gets equal expected count per batch.
    # More reliable than class weights in the loss, because imbalance is fixed
    # before the loss sees the data — the model cannot learn to ignore Down.
    labels_arr = train_ds.labels
    n_down = (labels_arr == 0).sum()
    n_up   = (labels_arr == 1).sum()
    w_down = 1.0 / n_down
    w_up   = 1.0 / n_up
    sample_weights = np.where(labels_arr == 0, w_down, w_up).astype(np.float32)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights),
        num_samples = len(sample_weights),
        replacement = True,
    )
    print(f'\nBalanced sampler — Down: {n_down:,}  Up: {n_up:,}')
    print(f'  w_down={w_down:.2e}  w_up={w_up:.2e}  (each gets ~50% per batch)')

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=True)

    all_preds   = []
    all_aucs    = []
    true_labels = None

    for run_id in range(N_RUNS):
        print(f"\n{'='*60}")
        print(f"  RUN {run_id+1} / {N_RUNS}")
        print(f"{'='*60}")
        preds, auc, labels = train_one_run(
            run_id, train_loader, val_loader, test_loader, device
        )
        all_preds.append(preds)
        all_aucs.append(auc)
        true_labels = labels

    avg_preds   = np.mean(all_preds, axis=0)
    true_labels = np.array(true_labels)
    final_auc   = roc_auc_score(true_labels, avg_preds)
    final_acc   = ((avg_preds > 0.5) == true_labels).mean()
    final_f1    = f1_score(true_labels, avg_preds > 0.5, average='macro', zero_division=0)
    final_brier = np.mean((avg_preds - true_labels) ** 2)

    print('\n' + '=' * 60)
    print('  FINAL ENSEMBLE  [Jiang et al. 2021 Baseline]')
    print(f'  AUC per run : {[f"{a:.4f}" for a in all_aucs]}')
    print(f'  Mean AUC    : {np.mean(all_aucs):.4f}  ±{np.std(all_aucs):.4f}')
    print(f'  Ensemble AUC: {final_auc:.4f}')
    print(f'  Ensemble Acc: {final_acc*100:.1f}%')
    print(f'  Ensemble F1 : {final_f1:.4f}')
    print(f'  Ensemble Brier: {final_brier:.4f}')
    print('=' * 60)

    out_path = f'{MODEL_DIR}/baseline_ensemble_auc{final_auc:.4f}.npy'
    np.save(out_path, avg_preds)
    print(f'  Predictions → {out_path}')
