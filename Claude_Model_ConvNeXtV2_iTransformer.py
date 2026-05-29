"""
Phase 2 — HybridModelV5 Training  (Windows + RTX 2070)
========================================================
Gleiche Datenbasis wie V3/V4:
  CSV:    ./thesis_data_claudev3/dataset_mapping.csv
  Images: ./thesis_data_claudev3/images/
  Walk-Forward: Train 2005-2018 | Val 2019-2020 | OOS 2021-2026

Verbesserungen gegenüber V4 (State-of-the-Art 2023-2024):

  1. ConvNeXt V2-Tiny  statt EfficientNet-B2
     Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with
             Masked Autoencoders"  (CVPR 2023)
     Gewinnt laut "Battle of the Backbones" (NeurIPS 2023) fast alle
     Fine-Tuning-Benchmarks gegen EfficientNet, SwinV2 und DINOv2.
     FCMAE-Pretraining auf IN-21k → stärkerer Domain-Transfer für
     chart-artige Bilder, die sich von ImageNet unterscheiden.
     Stages 0-2 eingefroren, nur Stage 3 trainierbar. Feature-Dim: 768.

  2. iTransformer  statt Standard-Transformer (Sequenzbranch)
     Paper: "iTransformer: Inverted Transformers Are Effective for
             Time Series Forecasting"  (ICLR 2024)
     Dreht das Token-Konzept um: Jede Feature-Variable (RSI, MACD,
     Bollinger, …) wird als eigener Token behandelt, nicht jeder
     Zeitschritt. Attention lernt cross-Indikator-Korrelationen
     statt Zeitschritt-zu-Zeitschritt-Muster — konzeptuell besser
     für multivariate Finanzzeitreihen der Länge 20.

  3. Asymmetric Focal Loss  statt symmetrischer Focal Loss
     Paper: "Asymmetric Loss For Multi-Label Classification"
             (ICCV 2021), adaptiert für binäre Klassifikation.
     gamma_neg=3.0 > gamma_pos=1.0: Stärkere Fokussierung auf
     einfache Fehler der Down-Klasse → bessere AUC bei Klassenimbalanz.

Voraussetzung:
  pip install timm
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
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
)

try:
    import timm
except ImportError:
    raise ImportError(
        "timm wird für ConvNeXt V2 benötigt.\n"
        "Installation: pip install timm"
    )

ssl._create_default_https_context = ssl._create_unverified_context

# ──────────────────────────────────────────────────────────────────────────────
# 1.  KONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH              = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_claudev4\dataset_mapping.csv'
MODEL_SAVE_PATH       = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_ConvNeXtV2_iTransformer\best_model.pth'
PREDICTIONS_SAVE_PATH = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_ConvNeXtV2_iTransformer\out_of_sample_predictions.csv'

WINDOW_SIZE  = 20
NUM_FEATURES = 25

BATCH_SIZE   = 128
EPOCHS       = 30
LR_HEAD      = 1e-4
LR_BACKBONE  = 1e-5
WEIGHT_DECAY = 3e-4
PATIENCE     = 7
USE_AMP      = True

CUTMIX_PROB  = 0.5
MIXUP_ALPHA  = 0.05
CUTMIX_ALPHA = 1.0

# Asymmetric Focal Loss Parameter
AFL_GAMMA_POS = 1.0   # gamma für Up-Samples  (schwieriger zu klassifizieren)
AFL_GAMMA_NEG = 2.0   # gamma für Down-Samples (stärker fokussieren)

TRAIN_END = '2018-12-31'
VAL_END   = '2020-12-31'

os.makedirs(r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_ConvNeXtV2_iTransformer', exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATASET  (RAM-optimiert: sequences vorparsed, kein CSV pro Worker)
# ──────────────────────────────────────────────────────────────────────────────
def preparse_csv(csv_path: str):
    """Liest CSV einmal im Hauptprozess, gibt kompakte numpy-Arrays zurück."""
    print("Lese CSV …")
    df = pd.read_csv(csv_path)

    print(f"Parse sequences ({len(df):,} Samples) …")
    seq_array = np.stack([
        np.fromstring(s, sep=',', dtype=np.float32).reshape(WINDOW_SIZE, NUM_FEATURES)
        for s in tqdm(df['numerical_seq'], ncols=80)
    ])  # (N, T, F)

    base_dir  = r'C:\Users\limga\Master Thesis\MasterThesis'
    img_paths = [
        os.path.normpath(os.path.join(base_dir, p.replace('\\', '/')))
        for p in df['image_path'].tolist()
    ]
    labels    = df['label'].values.astype(np.int64)
    dates     = df['date'].values

    del df
    return seq_array, img_paths, labels, dates


class ThesisHybridDataset(Dataset):
    """Leichtgewichtiger Dataset mit vorparsed numpy-Arrays."""
    def __init__(self, img_paths: list, seq_array: np.ndarray,
                 labels: np.ndarray, augment: bool = False):
        self.img_paths = img_paths
        self.seq_array = seq_array
        self.labels    = labels

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
class iTransformer(nn.Module):
    """
    iTransformer (ICLR 2024):
    Behandelt jede Feature-Variable als Token statt jeden Zeitschritt.

    Input:  (B, T=20, F=21)
    Transpose → (B, F=21, T=20)
    Proj:      Linear(T=20, d=128)  →  (B, 21, 128)
    Attention: über 21 Feature-Tokens  →  cross-Indikator-Korrelationen
    Pool:      Mean über Features  →  (B, 128)
    """
    def __init__(self, n_feat: int = NUM_FEATURES, seq_len: int = WINDOW_SIZE,
                 d: int = 128, heads: int = 4, layers: int = 2, drop: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(seq_len, d)
        enc          = nn.TransformerEncoderLayer(
            d, heads, d * 4, drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm    = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)   # (B, T, F) → (B, F, T)
        x = self.proj(x)         # (B, F, T) → (B, F, d)
        x = self.encoder(x)      # (B, F, d) → (B, F, d)
        return self.norm(x).mean(dim=1)   # pool über Features → (B, d)


class CrossModalFusion(nn.Module):
    """Cross-Attention zwischen CNN-Features und Sequenz-Features."""
    def __init__(self, img_d: int = 128, seq_d: int = 128, out_d: int = 128):
        super().__init__()
        self.q   = nn.Linear(img_d, seq_d)
        self.k   = nn.Linear(seq_d, seq_d)
        self.v   = nn.Linear(seq_d, seq_d)
        self.out = nn.Sequential(
            nn.Linear(img_d + seq_d, out_d),
            nn.BatchNorm1d(out_d), nn.GELU(), nn.Dropout(0.1))

    def forward(self, xi: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        a = F.scaled_dot_product_attention(
            self.q(xi).unsqueeze(1),
            self.k(xs).unsqueeze(1),
            self.v(xs).unsqueeze(1)).squeeze(1)
        return self.out(torch.cat([xi, a], dim=1))


class HybridModelV5(nn.Module):
    def __init__(self):
        super().__init__()

        # ── CNN Backbone: ConvNeXt V2-Tiny (CVPR 2023) ───────────────────
        # FCMAE-pretrained auf IN-21k → IN-1k
        # Stages 0-2 eingefroren (Stem+12 Blöcke), nur Stage 3 trainierbar
        # (Stage 3 = 3 Blöcke auf 7×7 spatial, 768ch → höchste semantische Features)
        # Freezing nach Modulreferenz statt Name für Robustheit
        try:
            self.backbone = timm.create_model(
                'convnextv2_tiny.fcmae_ft_in22k_in1k',
                pretrained=True, num_classes=0, global_pool='avg')
            print("  Backbone: ConvNeXt V2-Tiny (FCMAE + IN22k→IN1k)")
        except Exception:
            self.backbone = timm.create_model(
                'convnextv2_tiny', pretrained=True,
                num_classes=0, global_pool='avg')
            print("  Backbone: ConvNeXt V2-Tiny (IN1k fallback)")
        # output: (B, 768)

        # Freeze stem + stages 0, 1, 2  →  nur stage 3 + norm trainierbar
        for module in [self.backbone.stem,
                       self.backbone.stages[0],
                       self.backbone.stages[1],
                       self.backbone.stages[2]]:
            for p in module.parameters():
                p.requires_grad = False

        self.cnn_head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
        )

        # ── Sequence Branch: iTransformer (ICLR 2024) ────────────────────
        self.seq = iTransformer()   # output: (B, 128)

        # ── Fusion + Classifier ───────────────────────────────────────────
        self.fusion = CrossModalFusion()
        self.clf    = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, img: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        x = self.cnn_head(self.backbone(img))
        s = self.seq(seq)
        return self.clf(self.fusion(x, s))

    def param_groups(self):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params     = (list(self.cnn_head.parameters()) +
                           list(self.seq.parameters())      +
                           list(self.fusion.parameters())   +
                           list(self.clf.parameters()))
        return [
            {'params': backbone_params, 'lr': LR_BACKBONE, 'name': 'backbone'},
            {'params': head_params,     'lr': LR_HEAD,     'name': 'heads'},
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ASYMMETRIC FOCAL LOSS  (ICCV 2021)
# ──────────────────────────────────────────────────────────────────────────────
class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss für binäre Klassifikation.
    gamma_neg > gamma_pos → stärkere Fokussierung auf schwierige Down-Fehler.
    """
    def __init__(self, gamma_pos: float = AFL_GAMMA_POS,
                 gamma_neg: float = AFL_GAMMA_NEG,
                 weight: torch.Tensor = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma_pos       = gamma_pos
        self.gamma_neg       = gamma_neg
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce    = F.cross_entropy(logits, targets, weight=self.weight,
                                label_smoothing=self.label_smoothing, reduction='none')
        pt    = torch.exp(-ce)
        gamma = torch.where(targets == 1,
                            torch.full_like(ce, self.gamma_pos),
                            torch.full_like(ce, self.gamma_neg))
        return ((1 - pt) ** gamma * ce).mean()


def afl_mixup_loss(criterion, out, ya, yb, lam):
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
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)

    imgs_cut = imgs.clone()
    imgs_cut[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return imgs_cut, seqs, labels, labels[idx], lam_actual


def augment_batch(imgs, seqs, labels, device):
    if np.random.rand() < CUTMIX_PROB:
        return cutmix(imgs, seqs, labels, device=device)
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
    print("\n--- Phase 2: HybridModelV5  (ConvNeXt V2 + iTransformer + AFL) ---")

    # ── Device ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("Kein CUDA — CPU wird verwendet (sehr langsam).")
        device, use_amp = torch.device('cpu'), False
    else:
        device  = torch.device('cuda')
        use_amp = USE_AMP
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
        print(f"GPU:  {torch.cuda.get_device_name(0)}"
              f"  |  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"AMP: {'on' if use_amp else 'off'}   cuDNN Benchmark: on")

    if not os.path.exists(CSV_PATH):
        print(f"CSV nicht gefunden: {CSV_PATH}")
        return

    # ── CSV einmal lesen + Sequences vorparsed ────────────────────────────
    seq_array, img_paths, labels, dates = preparse_csv(CSV_PATH)

    # ── Walk-Forward Split ────────────────────────────────────────────────
    dt         = pd.to_datetime([str(d) for d in dates])
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

    # ── Normalisierung auf Train-Stats ────────────────────────────────────
    print("Normalisiere Sequences (Train-Stats) …")
    seq_mean  = seq_array[train_idx].mean(axis=(0, 1))
    seq_std   = seq_array[train_idx].std(axis=(0, 1)) + 1e-8
    seq_array = ((seq_array - seq_mean) / seq_std).astype(np.float32)

    # ── Datasets ──────────────────────────────────────────────────────────
    aug_ds  = ThesisHybridDataset(img_paths, seq_array, labels, augment=True)
    base_ds = ThesisHybridDataset(img_paths, seq_array, labels, augment=False)

    train_set = torch.utils.data.Subset(aug_ds,  train_idx)
    val_set   = torch.utils.data.Subset(base_ds, val_idx)
    test_set  = torch.utils.data.Subset(base_ds, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True,
                              persistent_workers=False, prefetch_factor=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Batch Size: {BATCH_SIZE}  |  Train-Workers: 2  |  Eval-Workers: 0")

    # ── Modell ────────────────────────────────────────────────────────────
    model  = HybridModelV5().to(device)
    total  = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter: {total/1e6:.2f}M total, {active/1e6:.2f}M trainierbar")

    try:
        import triton  # noqa
        model = torch.compile(model, backend='inductor')
        print("torch.compile(inductor) aktiviert")
    except ImportError:
        print("torch.compile übersprungen (Triton nicht verfügbar)")

    # ── Asymmetric Focal Loss mit Klassengewichten ────────────────────────
    train_labels = labels[train_idx]
    counts       = torch.bincount(torch.tensor(train_labels))
    print(f"\nKlassenverteilung — Down(0): {counts[0]:,}  Up(1): {counts[1]:,}")
    weights   = (1.0 / counts.float())
    weights   = (weights / weights.sum() * 2.0).to(device)
    criterion = AsymmetricFocalLoss(
        gamma_pos=AFL_GAMMA_POS, gamma_neg=AFL_GAMMA_NEG,
        weight=weights, label_smoothing=0.05)
    print(f"AFL: gamma_pos={AFL_GAMMA_POS}  gamma_neg={AFL_GAMMA_NEG}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.param_groups(),
        weight_decay=WEIGHT_DECAY,
    )

    # ── Scheduler ────────────────────────────────────────────────────────
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
                loss = afl_mixup_loss(criterion, out, ya, yb, lam)

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
