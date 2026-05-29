"""
run_image_comparison.py
=======================
Trains EfficientNet-B2 and ConvNeXt V2 on both image encodings:
  v3 → candlestick 5-panel charts (224×224 RGB, 21 features)
  v4 → multi-scale RGB (R=5d G=20d B=60d channels, 25 features)

Four runs in sequence:
  1. EfficientNet-B2   on v3_fixed  →  thesis_data_comparison/efficientnet_v3/
  2. EfficientNet-B2   on v4        →  thesis_data_comparison/efficientnet_v4/
  3. ConvNeXt V2-Tiny  on v3_fixed  →  thesis_data_comparison/convnext_v3/
  4. ConvNeXt V2-Tiny  on v4        →  thesis_data_comparison/convnext_v4/

Existing model files are NOT touched.
Run generate_v3_fixed_csv.py first if thesis_data_claudev3_fixed/ does not exist.
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

ssl._create_default_https_context = ssl._create_unverified_context

# ──────────────────────────────────────────────────────────────────────────────
# 1.  RUN CONFIGURATIONS
# ──────────────────────────────────────────────────────────────────────────────
BASE = r'C:\Users\limga\Master Thesis\MasterThesis'
COMP = os.path.join(BASE, 'thesis_data_comparison')

V3_CSV = os.path.join(BASE, 'thesis_data_claudev3_fixed', 'dataset_mapping.csv')
V4_CSV = os.path.join(BASE, 'thesis_data_claudev4',       'dataset_mapping.csv')

RUNS = [
    {
        'name':         'EfficientNet-v3',
        'model_type':   'efficientnet',
        'csv_path':     V3_CSV,
        'num_features': 21,
        'model_path':   os.path.join(COMP, 'efficientnet_v3', 'best_model.pth'),
        'preds_path':   os.path.join(COMP, 'efficientnet_v3', 'predictions.csv'),
    },
    {
        'name':         'EfficientNet-v4',
        'model_type':   'efficientnet',
        'csv_path':     V4_CSV,
        'num_features': 25,
        'model_path':   os.path.join(COMP, 'efficientnet_v4', 'best_model.pth'),
        'preds_path':   os.path.join(COMP, 'efficientnet_v4', 'predictions.csv'),
    },
    {
        'name':         'ConvNeXt-v3',
        'model_type':   'convnext',
        'csv_path':     V3_CSV,
        'num_features': 21,
        'model_path':   os.path.join(COMP, 'convnext_v3', 'best_model.pth'),
        'preds_path':   os.path.join(COMP, 'convnext_v3', 'predictions.csv'),
    },
    {
        'name':         'ConvNeXt-v4',
        'model_type':   'convnext',
        'csv_path':     V4_CSV,
        'num_features': 25,
        'model_path':   os.path.join(COMP, 'convnext_v4', 'best_model.pth'),
        'preds_path':   os.path.join(COMP, 'convnext_v4', 'predictions.csv'),
    },
]

# Shared hyperparameters
WINDOW_SIZE = 20
TRAIN_END   = '2018-12-31'
VAL_END     = '2020-12-31'

# EfficientNet-specific hyperparameters
EN_BATCH      = 256
EN_EPOCHS     = 30
EN_LR_HEAD    = 3e-4
EN_LR_BACK    = 3e-5
EN_WEIGHT_DECAY = 1e-4
EN_PATIENCE   = 7
EN_FOCAL_GAMMA = 2.0
EN_CUTMIX_PROB = 0.5
EN_MIXUP_ALPHA = 0.05
EN_CUTMIX_ALPHA = 1.0

# ConvNeXt-specific hyperparameters
CN_BATCH      = 128
CN_EPOCHS     = 30
CN_LR_HEAD    = 1e-4
CN_LR_BACK    = 1e-5
CN_WEIGHT_DECAY = 3e-4
CN_PATIENCE   = 7
CN_AFL_POS    = 1.0
CN_AFL_NEG    = 2.0
CN_CUTMIX_PROB = 0.5
CN_MIXUP_ALPHA = 0.05
CN_CUTMIX_ALPHA = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SHARED DATASET UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def preparse_csv(csv_path: str, num_features: int, window_size: int = WINDOW_SIZE):
    print("Reading CSV …")
    df = pd.read_csv(csv_path)
    print(f"Parsing sequences ({len(df):,} samples, {num_features} features) …")
    seq_array = np.stack([
        np.fromstring(s, sep=',', dtype=np.float32).reshape(window_size, num_features)
        for s in tqdm(df['numerical_seq'], ncols=80)
    ])
    img_paths = [
        os.path.normpath(os.path.join(BASE, p.replace('\\', '/')))
        for p in df['image_path'].tolist()
    ]
    labels = df['label'].values.astype(np.int64)
    dates  = df['date'].values
    del df
    return seq_array, img_paths, labels, dates


class ThesisHybridDataset(Dataset):
    def __init__(self, img_paths, seq_array, labels, augment=False):
        self.img_paths = img_paths
        self.seq_array = seq_array
        self.labels    = labels
        base_t = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        aug_t = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
        self.transform = transforms.Compose(aug_t if augment else base_t)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img), torch.from_numpy(self.seq_array[idx]), \
               torch.tensor(self.labels[idx], dtype=torch.long), idx


def evaluate(model, loader, device, use_amp):
    model.eval()
    targets, preds, probs, idxs = [], [], [], []
    with torch.no_grad():
        for imgs, seqs, labels, idx in loader:
            imgs = imgs.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            with autocast('cuda', enabled=use_amp):
                out = model(imgs, seqs)
            p = F.softmax(out.float(), dim=1)[:, 1]
            targets.extend(labels.cpu().numpy())
            preds.extend(out.float().argmax(1).cpu().numpy())
            probs.extend(p.cpu().numpy())
            idxs.extend(idx.cpu().numpy())
    auc = roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.5
    return dict(
        acc=accuracy_score(targets, preds),
        auc=auc,
        f1=f1_score(targets, preds, average='macro', zero_division=0),
        brier=brier_score_loss(targets, probs),
        targets=targets, preds=preds, probs=probs, idxs=idxs,
    )


def mixup(imgs, seqs, labels, alpha, device):
    lam = max(float(np.random.beta(alpha, alpha)), 1 - float(np.random.beta(alpha, alpha)))
    idx = torch.randperm(imgs.size(0), device=device)
    return lam*imgs + (1-lam)*imgs[idx], lam*seqs + (1-lam)*seqs[idx], labels, labels[idx], lam


def cutmix(imgs, seqs, labels, alpha, device):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=device)
    _, _, H, W = imgs.shape
    cut_w = int(W * math.sqrt(1 - lam));  cut_h = int(H * math.sqrt(1 - lam))
    cx = np.random.randint(W);            cy = np.random.randint(H)
    x1 = max(cx - cut_w//2, 0);          x2 = min(cx + cut_w//2, W)
    y1 = max(cy - cut_h//2, 0);          y2 = min(cy + cut_h//2, H)
    out = imgs.clone()
    out[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2-x1)*(y2-y1)/(W*H)
    return out, seqs, labels, labels[idx], lam_actual


# ──────────────────────────────────────────────────────────────────────────────
# 3.  EFFICIENTNET-B2  MODEL  (HybridModelV4)
# ──────────────────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels//reduction, bias=False), nn.ReLU(),
            nn.Linear(channels//reduction, channels, bias=False), nn.Sigmoid())
        self.sp = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ch(x).view(x.size(0), -1, 1, 1)
        return x * self.sp(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1))


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=100, drop=0.1):
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
    def __init__(self, n_feat, d=128, heads=8, layers=3, drop=0.1):
        super().__init__()
        self.proj    = nn.Linear(n_feat, d)
        self.pos     = PositionalEncoding(d, drop=drop)
        enc          = nn.TransformerEncoderLayer(d, heads, d*4, drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)

    def forward(self, x):
        return self.encoder(self.pos(self.proj(x)))[:, -1]


class CrossModalFusion(nn.Module):
    def __init__(self, img_d=128, seq_d=128, out_d=128):
        super().__init__()
        self.q   = nn.Linear(img_d, seq_d)
        self.k   = nn.Linear(seq_d, seq_d)
        self.v   = nn.Linear(seq_d, seq_d)
        self.out = nn.Sequential(
            nn.Linear(img_d + seq_d, out_d), nn.BatchNorm1d(out_d), nn.GELU(), nn.Dropout(0.1))

    def forward(self, xi, xs):
        a = F.scaled_dot_product_attention(
            self.q(xi).unsqueeze(1), self.k(xs).unsqueeze(1), self.v(xs).unsqueeze(1)).squeeze(1)
        return self.out(torch.cat([xi, a], 1))


class HybridModelV4(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        base      = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.cnn  = base.features
        self.cbam = CBAM(1408)
        for name, p in self.cnn.named_parameters():
            block = int(name.split('.')[0]) if name[0].isdigit() else -1
            p.requires_grad = block >= 4
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.cnn_head = nn.Sequential(
            nn.Linear(1408, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1))
        self.seq    = SequenceTransformer(n_feat=n_feat)
        self.fusion = CrossModalFusion()
        self.clf    = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, img, seq):
        x = self.pool(self.cbam(self.cnn(img))).flatten(1)
        return self.clf(self.fusion(self.cnn_head(x), self.seq(seq)))

    def param_groups(self, lr_backbone, lr_head):
        backbone = list(self.cnn.parameters()) + list(self.cbam.parameters())
        heads    = (list(self.pool.parameters()) + list(self.cnn_head.parameters()) +
                    list(self.seq.parameters())  + list(self.fusion.parameters()) +
                    list(self.clf.parameters()))
        return [
            {'params': [p for p in backbone if p.requires_grad], 'lr': lr_backbone, 'name': 'backbone'},
            {'params': heads, 'lr': lr_head, 'name': 'heads'},
        ]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CONVNEXT V2-TINY  MODEL  (HybridModelV5)
# ──────────────────────────────────────────────────────────────────────────────
class iTransformer(nn.Module):
    def __init__(self, n_feat, seq_len=WINDOW_SIZE, d=128, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.proj    = nn.Linear(seq_len, d)
        enc          = nn.TransformerEncoderLayer(d, heads, d*4, drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm    = nn.LayerNorm(d)

    def forward(self, x):
        x = self.proj(x.transpose(1, 2))   # (B,T,F) → (B,F,T) → (B,F,d)
        return self.norm(self.encoder(x)).mean(dim=1)


class HybridModelV5(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                'convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0, global_pool='avg')
            print("  Backbone: ConvNeXt V2-Tiny (FCMAE + IN22k→IN1k)")
        except Exception:
            self.backbone = timm.create_model(
                'convnextv2_tiny', pretrained=True, num_classes=0, global_pool='avg')
            print("  Backbone: ConvNeXt V2-Tiny (IN1k fallback)")
        for module in [self.backbone.stem, self.backbone.stages[0],
                       self.backbone.stages[1], self.backbone.stages[2]]:
            for p in module.parameters():
                p.requires_grad = False
        self.cnn_head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1))
        self.seq    = iTransformer(n_feat=n_feat)
        self.fusion = CrossModalFusion()
        self.clf    = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, img, seq):
        return self.clf(self.fusion(self.cnn_head(self.backbone(img)), self.seq(seq)))

    def param_groups(self, lr_backbone, lr_head):
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        heads    = (list(self.cnn_head.parameters()) + list(self.seq.parameters()) +
                    list(self.fusion.parameters())   + list(self.clf.parameters()))
        return [
            {'params': backbone, 'lr': lr_backbone, 'name': 'backbone'},
            {'params': heads,    'lr': lr_head,     'name': 'heads'},
        ]


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma_pos       = gamma_pos
        self.gamma_neg       = gamma_neg
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets, weight=self.weight,
                                label_smoothing=self.label_smoothing, reduction='none')
        pt    = torch.exp(-ce)
        gamma = torch.where(targets == 1,
                            torch.full_like(ce, self.gamma_pos),
                            torch.full_like(ce, self.gamma_neg))
        return ((1 - pt) ** gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  TRAINING FUNCTION  (shared, parameterised per run)
# ──────────────────────────────────────────────────────────────────────────────
def train_one_run(cfg: dict):
    name        = cfg['name']
    model_type  = cfg['model_type']
    csv_path    = cfg['csv_path']
    n_feat      = cfg['num_features']
    model_path  = cfg['model_path']
    preds_path  = cfg['preds_path']

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  RUN: {name}")
    print(f"  Model  : {model_type.upper()}  |  Features: {n_feat}")
    print(f"  CSV    : {csv_path}")
    print(f"  Output : {os.path.dirname(model_path)}")
    print(f"{'='*70}")

    if not os.path.exists(csv_path):
        print(f"  ERROR: CSV not found — {csv_path}")
        print(f"  Run generate_v3_fixed_csv.py first if this is a v3 run.")
        return False

    # Device
    if torch.cuda.is_available():
        device  = torch.device('cuda')
        use_amp = True
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False
        print(f"\nGPU: {torch.cuda.get_device_name(0)}"
              f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        device, use_amp = torch.device('cpu'), False
        print("No CUDA — using CPU (slow).")

    # Data
    seq_array, img_paths, labels, dates = preparse_csv(csv_path, n_feat)

    dt         = pd.to_datetime([str(d) for d in dates])
    train_mask = dt <= pd.Timestamp(TRAIN_END)
    val_mask   = (dt > pd.Timestamp(TRAIN_END)) & (dt <= pd.Timestamp(VAL_END))
    test_mask  = dt > pd.Timestamp(VAL_END)
    train_idx  = np.where(train_mask)[0]
    val_idx    = np.where(val_mask)[0]
    test_idx   = np.where(test_mask)[0]

    print(f"\nWalk-Forward Split:")
    print(f"  Train : {len(train_idx):>7,}  (2005–2018)")
    print(f"  Val   : {len(val_idx):>7,}  (2019–2020)")
    print(f"  OOS   : {len(test_idx):>7,}  (2021–2026)")

    # Normalise sequences using train statistics only
    seq_mean  = seq_array[train_idx].mean(axis=(0, 1))
    seq_std   = seq_array[train_idx].std(axis=(0, 1)) + 1e-8
    seq_array = ((seq_array - seq_mean) / seq_std).astype(np.float32)

    aug_ds  = ThesisHybridDataset(img_paths, seq_array, labels, augment=True)
    base_ds = ThesisHybridDataset(img_paths, seq_array, labels, augment=False)

    if model_type == 'efficientnet':
        batch = EN_BATCH;  n_workers_train = 4
    else:
        batch = CN_BATCH;  n_workers_train = 2

    train_loader = DataLoader(torch.utils.data.Subset(aug_ds,  train_idx),
                              batch_size=batch, shuffle=True,
                              num_workers=n_workers_train, pin_memory=True,
                              persistent_workers=(n_workers_train > 0), prefetch_factor=4 if n_workers_train >= 4 else 2)
    val_loader   = DataLoader(torch.utils.data.Subset(base_ds, val_idx),
                              batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(torch.utils.data.Subset(base_ds, test_idx),
                              batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    # Model + loss + optimiser
    if model_type == 'efficientnet':
        model     = HybridModelV4(n_feat=n_feat).to(device)
        criterion = FocalLoss(gamma=EN_FOCAL_GAMMA, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.param_groups(EN_LR_BACK, EN_LR_HEAD),
                                      weight_decay=EN_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6)
        epochs    = EN_EPOCHS
        patience  = EN_PATIENCE
        cutmix_prob  = EN_CUTMIX_PROB
        mixup_alpha  = EN_MIXUP_ALPHA
        cutmix_alpha = EN_CUTMIX_ALPHA

        def loss_fn(criterion, out, ya, yb, lam):
            return lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)

    else:  # convnext
        model = HybridModelV5(n_feat=n_feat).to(device)
        train_labels = labels[train_idx]
        counts   = torch.bincount(torch.tensor(train_labels))
        weights  = (1.0 / counts.float())
        weights  = (weights / weights.sum() * 2.0).to(device)
        criterion = AsymmetricFocalLoss(gamma_pos=CN_AFL_POS, gamma_neg=CN_AFL_NEG,
                                        weight=weights, label_smoothing=0.05)
        print(f"AFL: gamma_pos={CN_AFL_POS}  gamma_neg={CN_AFL_NEG}"
              f"  Down(0): {counts[0]:,}  Up(1): {counts[1]:,}")
        optimizer = torch.optim.AdamW(model.param_groups(CN_LR_BACK, CN_LR_HEAD),
                                      weight_decay=CN_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6)
        epochs    = CN_EPOCHS
        patience  = CN_PATIENCE
        cutmix_prob  = CN_CUTMIX_PROB
        mixup_alpha  = CN_MIXUP_ALPHA
        cutmix_alpha = CN_CUTMIX_ALPHA

        def loss_fn(criterion, out, ya, yb, lam):
            return lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)

    total  = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {total/1e6:.2f}M total, {active/1e6:.2f}M trainable")

    try:
        import triton  # noqa
        model = torch.compile(model, backend='inductor')
        print("torch.compile(inductor) active")
    except ImportError:
        pass

    scaler = GradScaler('cuda', enabled=use_amp)
    best_val_auc, no_improve = 0.0, 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for imgs, seqs, batch_labels, _ in tqdm(
                train_loader, desc=f"[{name}] Epoch {epoch+1}/{epochs}", ncols=100):
            imgs         = imgs.to(device, non_blocking=True)
            seqs         = seqs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            if np.random.rand() < cutmix_prob:
                imgs_m, seqs_m, ya, yb, lam = cutmix(imgs, seqs, batch_labels, cutmix_alpha, device)
            else:
                imgs_m, seqs_m, ya, yb, lam = mixup(imgs, seqs, batch_labels, mixup_alpha, device)

            with autocast('cuda', enabled=use_amp):
                out  = model(imgs_m, seqs_m)
                loss = loss_fn(criterion, out, ya, yb, lam)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        epoch_time = time.time() - t0
        vram = torch.cuda.memory_allocated() / 1e9 if use_amp else 0.0

        val_m  = evaluate(model, val_loader,  device, use_amp)
        test_m = evaluate(model, test_loader, device, use_amp)
        scheduler.step(val_m['auc'])

        lr_h = optimizer.param_groups[1]['lr']
        lr_b = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}  Loss: {running_loss/len(train_loader):.4f}"
              f"  LR-Head: {lr_h:.2e}  LR-Back: {lr_b:.2e}"
              f"  {epoch_time:.0f}s  VRAM: {vram:.1f}GB")
        print(f"  [VAL]  AUC: {val_m['auc']:.4f}  Acc: {val_m['acc']*100:.2f}%"
              f"  F1: {val_m['f1']:.4f}  Brier: {val_m['brier']:.4f}")
        print(f"  [OOS]  AUC: {test_m['auc']:.4f}  Acc: {test_m['acc']*100:.2f}%"
              f"  F1: {test_m['f1']:.4f}  Brier: {test_m['brier']:.4f}")

        if val_m['auc'] > best_val_auc:
            best_val_auc = val_m['auc']
            no_improve   = 0
            torch.save(model.state_dict(), model_path)
            oos_idxs = test_m['idxs']
            pd.DataFrame({
                'image_path':      [img_paths[i] for i in oos_idxs],
                'date':            dates[oos_idxs],
                'label':           labels[oos_idxs],
                'Predicted_Class': test_m['preds'],
                'Prob_Up':         test_m['probs'],
            }).to_csv(preds_path, index=False)
            print(f"  >> New best Val-AUC: {best_val_auc:.4f}  (OOS: {test_m['auc']:.4f})  saved.")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                print("  Early stopping.")
                break

    print(f"\n[{name}] Done.  Best Val-AUC: {best_val_auc:.4f}")
    print(f"  Model : {model_path}")
    print(f"  Preds : {preds_path}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_total = time.time()
    results = []

    for i, cfg in enumerate(RUNS):
        t_run = time.time()
        print(f"\n\n{'#'*70}")
        print(f"  RUN {i+1}/{len(RUNS)}: {cfg['name']}")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")
        ok = train_one_run(cfg)
        elapsed = (time.time() - t_run) / 3600
        results.append((cfg['name'], ok, elapsed))
        print(f"\n  Run {i+1} finished in {elapsed:.2f}h  ({'OK' if ok else 'FAILED'})")

    total_h = (time.time() - t_total) / 3600
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON RUN COMPLETE  ({total_h:.2f}h total)")
    print(f"  Output directory: {COMP}")
    print(f"{'='*70}")
    for name, ok, h in results:
        status = 'OK  ' if ok else 'FAIL'
        print(f"  [{status}]  {name:<25}  {h:.2f}h")
    print(f"{'='*70}")
    print(f"\nNext step: run performance_analysis.py pointing at {COMP}")
