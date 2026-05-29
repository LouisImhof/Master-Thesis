"""
Phase 2 — HybridModelV4 Training  (Windows + RTX 2070)
========================================================
Gleiche Datenbasis wie V3:
  CSV:    ./thesis_data_claudev3/dataset_mapping.csv
  Images: ./thesis_data_claudev3/images/
  Walk-Forward: Train 2005-2018 | Val 2019-2020 | OOS 2021-2026

Verbesserungen gegenüber V3:
  1. EfficientNet-B2 statt B0  (1408 Ch, mehr Kapazität)
  2. Transformer: d=128, heads=8, layers=3  (statt 64/4/2)
  3. CrossModalFusion angepasst auf neue Dimensionen
  4. CutMix + Mixup  (50/50 zufällig, statt nur Mixup)
  5. Focal Loss  (gamma=2.0, fokussiert auf schwierige Samples)
  6. ReduceLROnPlateau  (mode=max, factor=0.5, patience=4)
  7. 30 Epochs (statt 20), Patience=7

RAM-Optimierungen:
  - Sequences werden EINMAL im Hauptprozess vorparsed (numpy float32)
  - Dataset bekommt kompakte Arrays statt 1-GB-CSV-Kopie pro Worker
  - Val/Test-Loader: num_workers=0 (kein paralleler Overhead)
  - Train-Loader: num_workers=4 für GPU-Prefetching
"""

import os
import math
import ssl
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
)

ssl._create_default_https_context = ssl._create_unverified_context

# ──────────────────────────────────────────────────────────────────────────────
# 1.  KONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH              = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_claudev4\dataset_mapping.csv'
MODEL_SAVE_PATH       = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_EfficientNetB2_Transformer\best_model.pth'
PREDICTIONS_SAVE_PATH = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_EfficientNetB2_Transformer\out_of_sample_predictions.csv'

WINDOW_SIZE  = 20
NUM_FEATURES = 25

BATCH_SIZE    = 256
EPOCHS        = 30
LR_HEAD       = 3e-4
LR_BACKBONE   = 3e-5
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7
USE_AMP       = True

CUTMIX_PROB   = 0.5   # Wahrscheinlichkeit CutMix (sonst Mixup)
MIXUP_ALPHA   = 0.05
CUTMIX_ALPHA  = 1.0
FOCAL_GAMMA   = 2.0

TRAIN_END = '2018-12-31'
VAL_END   = '2020-12-31'

os.makedirs(r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_EfficientNetB2_Transformer', exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATASET  (RAM-optimiert: sequences vorparsed, kein CSV pro Worker)
# ──────────────────────────────────────────────────────────────────────────────
def preparse_csv(csv_path: str):
    """
    Liest CSV einmal im Hauptprozess.
    Gibt kompakte numpy-Arrays zurück — kein String-Parsing in Workers.
    """
    print("Lese CSV …")
    df = pd.read_csv(csv_path)

    print(f"Pars sequences ({len(df):,} Samples) …")
    seq_array = np.stack([
        np.fromstring(s, sep=',', dtype=np.float32).reshape(WINDOW_SIZE, NUM_FEATURES)
        for s in tqdm(df['numerical_seq'], ncols=80)
    ])  # shape: (N, WINDOW, FEATURES)

    base_dir  = r'C:\Users\limga\Master Thesis\MasterThesis'
    img_paths = [
        os.path.normpath(os.path.join(base_dir, p.replace('\\', '/')))
        for p in df['image_path'].tolist()
    ]
    labels    = df['label'].values.astype(np.int64)
    dates     = df['date'].values

    del df   # CSV-DataFrame sofort freigeben
    return seq_array, img_paths, labels, dates


class ThesisHybridDataset(Dataset):
    """
    Leichtgewichtiger Dataset: bekommt vorparsed numpy-Arrays.
    Jeder Worker sieht nur kompakte float32-Arrays (kein CSV, kein String-Parsing).
    """
    def __init__(self, img_paths: list, seq_array: np.ndarray,
                 labels: np.ndarray, augment: bool = False):
        self.img_paths = img_paths
        self.seq_array = seq_array   # (N, WINDOW, FEATURES) — schon normalisiert
        self.labels    = labels
        self.augment   = augment

        base_t = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        aug_t = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
        self.transform = transforms.Compose(aug_t if augment else base_t)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img        = Image.open(self.img_paths[idx]).convert('RGB')
        img_tensor = self.transform(img)
        seq_tensor = torch.from_numpy(self.seq_array[idx])
        label      = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, seq_tensor, label, idx


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MODELL
# ──────────────────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid(),
        )
        self.sp = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ch(x).view(x.size(0), -1, 1, 1)
        return x * self.sp(torch.cat([x.mean(1, keepdim=True),
                                      x.max(1, keepdim=True)[0]], 1))


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 100, drop: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class SequenceTransformer(nn.Module):
    """V4: d=128, heads=8, layers=3  (V3 war: 64/4/2)"""
    def __init__(self, n_feat: int = NUM_FEATURES, d: int = 128,
                 heads: int = 8, layers: int = 3, drop: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(n_feat, d)
        self.pos     = PositionalEncoding(d, drop=drop)
        enc          = nn.TransformerEncoderLayer(
            d, heads, d * 4, drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)

    def forward(self, x):
        return self.encoder(self.pos(self.proj(x)))[:, -1]


class CrossModalFusion(nn.Module):
    """V4: img_d=128, seq_d=128, out_d=128  (V3 war: 128/64/64)"""
    def __init__(self, img_d: int = 128, seq_d: int = 128, out_d: int = 128):
        super().__init__()
        self.q   = nn.Linear(img_d, seq_d)
        self.k   = nn.Linear(seq_d, seq_d)
        self.v   = nn.Linear(seq_d, seq_d)
        self.out = nn.Sequential(
            nn.Linear(img_d + seq_d, out_d),
            nn.BatchNorm1d(out_d), nn.GELU(), nn.Dropout(0.1))

    def forward(self, xi, xs):
        a = F.scaled_dot_product_attention(
            self.q(xi).unsqueeze(1),
            self.k(xs).unsqueeze(1),
            self.v(xs).unsqueeze(1)).squeeze(1)
        return self.out(torch.cat([xi, a], 1))


class HybridModelV4(nn.Module):
    def __init__(self):
        super().__init__()

        # ── CNN Backbone: EfficientNet-B2  (1408 Ch statt 1280) ──────────
        base      = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.cnn  = base.features      # output: (B, 1408, 7, 7)
        self.cbam = CBAM(1408)

        for name, p in self.cnn.named_parameters():
            block = int(name.split('.')[0]) if name[0].isdigit() else -1
            p.requires_grad = block >= 4

        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.cnn_head = nn.Sequential(
            nn.Linear(1408, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256,  128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
        )

        # ── Sequence Branch: größerer Transformer ─────────────────────────
        self.seq = SequenceTransformer()   # output: 128

        # ── Fusion + Classifier ───────────────────────────────────────────
        self.fusion = CrossModalFusion()   # img_d=128, seq_d=128 → 128
        self.clf    = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, img, seq):
        x = self.pool(self.cbam(self.cnn(img))).flatten(1)
        x = self.cnn_head(x)
        s = self.seq(seq)
        return self.clf(self.fusion(x, s))

    def param_groups(self):
        backbone = list(self.cnn.parameters()) + list(self.cbam.parameters())
        heads    = (list(self.pool.parameters())    +
                    list(self.cnn_head.parameters()) +
                    list(self.seq.parameters())      +
                    list(self.fusion.parameters())   +
                    list(self.clf.parameters()))
        return [
            {'params': [p for p in backbone if p.requires_grad],
             'lr': LR_BACKBONE, 'name': 'backbone'},
            {'params': heads,
             'lr': LR_HEAD,     'name': 'heads'},
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FOCAL LOSS
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss: fokussiert auf schwierige Samples, reduziert Einfluss
    einfacher Samples. gamma=2 ist Standard aus dem Original-Paper."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce  = F.cross_entropy(logits, targets, weight=self.weight,
                              label_smoothing=self.label_smoothing, reduction='none')
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def focal_mixup_loss(criterion, out, ya, yb, lam):
    return lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  AUGMENTATION: MIXUP + CUTMIX
# ──────────────────────────────────────────────────────────────────────────────
def mixup(imgs, seqs, labels, alpha=MIXUP_ALPHA, device='cuda'):
    lam = max(float(np.random.beta(alpha, alpha)),
              1 - float(np.random.beta(alpha, alpha)))
    idx = torch.randperm(imgs.size(0), device=device)
    return (lam * imgs + (1 - lam) * imgs[idx],
            lam * seqs + (1 - lam) * seqs[idx],
            labels, labels[idx], lam)


def cutmix(imgs, seqs, labels, alpha=CUTMIX_ALPHA, device='cuda'):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=device)

    _, _, H, W = imgs.shape
    cut_ratio  = math.sqrt(1 - lam)
    cut_w      = int(W * cut_ratio)
    cut_h      = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    imgs_cut = imgs.clone()
    imgs_cut[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]

    # Lambda anpassen auf tatsächlich ausgeschnittene Fläche
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return imgs_cut, seqs, labels, labels[idx], lam_actual


def augment_batch(imgs, seqs, labels, device):
    if np.random.rand() < CUTMIX_PROB:
        return cutmix(imgs, seqs, labels, device=device)
    else:
        return mixup(imgs, seqs, labels, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  EVALUATION HELPER
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, use_amp):
    model.eval()
    all_targets, all_preds, all_probs, all_idxs = [], [], [], []
    with torch.no_grad():
        for imgs, seqs, labels, idxs in loader:
            imgs = imgs.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            with autocast('cuda', enabled=use_amp):
                out = model(imgs, seqs)
            probs = F.softmax(out.float(), dim=1)[:, 1]
            preds = out.float().argmax(dim=1)
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_idxs.extend(idxs.cpu().numpy())

    acc   = accuracy_score(all_targets, all_preds)
    f1    = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    brier = brier_score_loss(all_targets, all_probs)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5
    return dict(acc=acc, auc=auc, f1=f1, brier=brier,
                targets=all_targets, preds=all_preds,
                probs=all_probs, idxs=all_idxs)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def train_and_evaluate():
    print("\n--- Phase 2: HybridModelV4  (EfficientNet-B2 + CutMix + FocalLoss) ---")

    # ── Device ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("Kein CUDA — CPU wird verwendet (sehr langsam).")
        device, use_amp = torch.device('cpu'), False
    else:
        device  = torch.device('cuda')
        use_amp = USE_AMP
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False
        print(f"GPU:  {torch.cuda.get_device_name(0)}"
              f"  |  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"AMP: {'on' if use_amp else 'off'}   cuDNN Benchmark: on")

    if not os.path.exists(CSV_PATH):
        print(f"CSV nicht gefunden: {CSV_PATH}")
        return

    # ── CSV einmal lesen + Sequences vorparsed ────────────────────────────
    seq_array, img_paths, labels, dates = preparse_csv(CSV_PATH)

    # ── Walk-Forward Split (nach Datum) ───────────────────────────────────
    dt        = pd.to_datetime([str(d) for d in dates])
    train_mask = dt <= pd.Timestamp(TRAIN_END)
    val_mask   = (dt > pd.Timestamp(TRAIN_END)) & (dt <= pd.Timestamp(VAL_END))
    test_mask  = dt > pd.Timestamp(VAL_END)

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]

    print(f"\nWalk-Forward Split:")
    print(f"  Train:      {len(train_idx):>7,}  (2005 – {TRAIN_END[:4]})")
    print(f"  Validation: {len(val_idx):>7,}  ({int(TRAIN_END[:4])+1} – {VAL_END[:4]})")
    print(f"  Test (OOS): {len(test_idx):>7,}  ({int(VAL_END[:4])+1} – 2026)")

    # ── Normalisierung NUR auf Train-Split fitten, dann alles normalisieren
    print("Normalisiere Sequences (Train-Stats) …")
    seq_mean = seq_array[train_idx].mean(axis=(0, 1))   # (FEATURES,)
    seq_std  = seq_array[train_idx].std(axis=(0, 1)) + 1e-8
    seq_array = ((seq_array - seq_mean) / seq_std).astype(np.float32)

    # ── Datasets: teilen dieselben Arrays (kein RAM-Duplizieren im Hauptprozess)
    aug_ds  = ThesisHybridDataset(img_paths, seq_array, labels, augment=True)
    base_ds = ThesisHybridDataset(img_paths, seq_array, labels, augment=False)

    train_set = torch.utils.data.Subset(aug_ds,  train_idx)
    val_set   = torch.utils.data.Subset(base_ds, val_idx)
    test_set  = torch.utils.data.Subset(base_ds, test_idx)

    # Train: 2 Workers für GPU-Prefetching
    # Val/Test: kein Worker-Overhead (werden sequenziell aufgerufen)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Batch Size: {BATCH_SIZE}  |  Train-Workers: 4  |  Eval-Workers: 0")

    # ── Modell ────────────────────────────────────────────────────────────
    model = HybridModelV4().to(device)
    total  = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter: {total/1e6:.2f}M total, {active/1e6:.2f}M trainierbar")

    try:
        import triton  # noqa
        model = torch.compile(model, backend='inductor')
        print("torch.compile(inductor) aktiviert")
    except ImportError:
        print("torch.compile übersprungen (Triton nicht verfügbar)")

    # ── Focal Loss (no separate class weights — focal loss handles imbalance) ──
    train_labels = labels[train_idx]
    counts       = torch.bincount(torch.tensor(train_labels))
    print(f"\nKlassenverteilung — Down(0): {counts[0]:,}  Up(1): {counts[1]:,}")
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=None, label_smoothing=0.05)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.param_groups(),
        weight_decay=WEIGHT_DECAY,
    )

    # ── Scheduler: ReduceLROnPlateau (einfach, keine Interferenz) ────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6)

    scaler = GradScaler('cuda', enabled=use_amp)

    # ── Training Loop ─────────────────────────────────────────────────────
    best_val_auc, no_improve = 0.0, 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        t0           = time.time()

        for imgs, seqs, batch_labels, _ in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
                ncols=100):
            imgs         = imgs.to(device, non_blocking=True)
            seqs         = seqs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            imgs_m, seqs_m, ya, yb, lam = augment_batch(
                imgs, seqs, batch_labels, device=device)

            with autocast('cuda', enabled=use_amp):
                out  = model(imgs_m, seqs_m)
                loss = focal_mixup_loss(criterion, out, ya, yb, lam)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_time = time.time() - t0
        vram       = torch.cuda.memory_allocated() / 1e9 if use_amp else 0.0

        # ── Validation ───────────────────────────────────────────────────
        val_m  = evaluate(model, val_loader,  device, use_amp)
        test_m = evaluate(model, test_loader, device, use_amp)

        scheduler.step(val_m['auc'])

        lr_head = optimizer.param_groups[1]['lr']
        lr_cnn  = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}  |  Loss: {running_loss/len(train_loader):.4f}"
              f"  LR-Head: {lr_head:.2e}  LR-CNN: {lr_cnn:.2e}"
              f"  Zeit: {epoch_time:.0f}s  VRAM: {vram:.1f}GB")
        print(f"  [VAL 2019-20]  Acc: {val_m['acc']*100:.2f}%"
              f"  AUC: {val_m['auc']:.4f}"
              f"  F1: {val_m['f1']:.4f}"
              f"  Brier: {val_m['brier']:.4f}")
        print(f"  [OOS 2021-26]  Acc: {test_m['acc']*100:.2f}%"
              f"  AUC: {test_m['auc']:.4f}"
              f"  F1: {test_m['f1']:.4f}"
              f"  Brier: {test_m['brier']:.4f}")

        # ── Checkpoint ───────────────────────────────────────────────────
        if val_m['auc'] > best_val_auc:
            best_val_auc = val_m['auc']
            no_improve   = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            oos_idxs = test_m['idxs']
            pred_df  = pd.DataFrame({
                'image_path':      [img_paths[i] for i in oos_idxs],
                'date':            dates[oos_idxs],
                'label':           labels[oos_idxs],
                'Predicted_Class': test_m['preds'],
                'Prob_Up':         test_m['probs'],
            })
            pred_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)

            print(f"  Neues bestes Val-AUC {best_val_auc:.4f}"
                  f"  (OOS: {test_m['auc']:.4f}) — gespeichert.")
        else:
            no_improve += 1
            print(f"  Keine Val-Verbesserung seit {no_improve}/{PATIENCE}.")
            if no_improve >= PATIENCE:
                print("  Early Stopping.")
                break

    print(f"\nTraining abgeschlossen.")
    print(f"  Bestes Val-AUC   (2019-20): {best_val_auc:.4f}")
    print(f"  Modell:          {MODEL_SAVE_PATH}")
    print(f"  OOS Predictions: {PREDICTIONS_SAVE_PATH}")


if __name__ == '__main__':
    train_and_evaluate()
