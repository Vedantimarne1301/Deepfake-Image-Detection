"""
train.py — Deepfake Detector Training (PyTorch + RTX 4050 / CUDA)
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR  = "C:\\Users\\VEDANTI\\OneDrive\\Desktop\\projects\\deepfake detection\\real_vs_fake\\real-vs-fake"
MODEL_PATH   = "model/deepfake_detector.pth"
IMG_SIZE     = 224
BATCH_SIZE   = 64
EPOCHS_HEAD  = 10
EPOCHS_FINE  = 10
NUM_WORKERS  = 2      
# ────────────────────────────────────────────────────────────────────────────



# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

# ── Transforms ───────────────────────────────────────────────────────────────
_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

# ── Datasets ─────────────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "valid"), transform=val_tf)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=False
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=False
)

print(f"Class map    : {train_ds.class_to_idx}")
print(f"Train images : {len(train_ds)}")
print(f"Val images   : {len(val_ds)}")

FAKE_IDX = train_ds.class_to_idx.get("fake", 1)
print(f"FAKE index   : {FAKE_IDX}")

# ── Prefetch helper — keeps GPU fed while CPU loads next batch ────────────────
class CUDAPrefetcher:
    """Asynchronously moves batches to GPU so the GPU never waits for CPU."""
    def __init__(self, loader, dev):
        self.loader = loader
        self.dev    = dev

    def __iter__(self):
        stream = torch.cuda.Stream()
        first  = True
        for next_imgs, next_lbls in self.loader:
            with torch.cuda.stream(stream):
                next_imgs = next_imgs.to(self.dev, non_blocking=True)
                next_lbls = next_lbls.to(self.dev, non_blocking=True)
            if not first:
                yield imgs, lbls
            else:
                first = False
            torch.cuda.current_stream().wait_stream(stream)
            imgs, lbls = next_imgs, next_lbls
        yield imgs, lbls

    def __len__(self):
        return len(self.loader)

# ── Model factory ────────────────────────────────────────────────────────────
def build_model(freeze_base=True):
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if freeze_base:
        for p in net.parameters():
            p.requires_grad = False
    in_f = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return net.to(device)

# ── Train / val loop ─────────────────────────────────────────────────────────
def run(model, optimizer, scheduler, epochs, tag):
    criterion    = nn.BCEWithLogitsLoss()
    best_acc     = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # Use prefetcher only on CUDA
    use_prefetch = (device.type == "cuda")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            model.train() if phase == "train" else model.eval()
            loss_sum, correct, total = 0.0, 0, 0

            iterator = CUDAPrefetcher(loader, device) if use_prefetch else loader

            for imgs, lbls in iterator:
                # imgs/lbls already on device when prefetching
                if not use_prefetch:
                    imgs = imgs.to(device, non_blocking=True)
                    lbls = lbls.to(device, non_blocking=True)

                lbls = lbls.float()
                if FAKE_IDX == 0:
                    lbls = 1.0 - lbls

                with torch.set_grad_enabled(phase == "train"):
                    out  = model(imgs).squeeze(1)
                    loss = criterion(out, lbls)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    preds    = (torch.sigmoid(out) >= 0.5).float()
                    correct  += (preds == lbls).sum().item()
                    total    += lbls.size(0)
                    loss_sum += loss.item() * lbls.size(0)

            ep_loss = loss_sum / total
            ep_acc  = correct  / total

            if phase == "val":
                scheduler.step(ep_loss)
                if ep_acc > best_acc:
                    best_acc     = ep_acc
                    best_weights = copy.deepcopy(model.state_dict())

        elapsed = time.time() - t0
        print(f"[{tag}] {epoch:02d}/{epochs}  "
              f"val_acc={ep_acc:.4f}  val_loss={ep_loss:.4f}  "
              f"({elapsed:.1f}s)")

    model.load_state_dict(best_weights)
    print(f"  Best val acc: {best_acc:.4f}")
    return model


# ── Entry point (required on Windows) ────────────────────────────────────────
if __name__ == '__main__':

    # Phase 1 — head only
    print("\n[Phase 1] Training head only ...")
    model = build_model(freeze_base=True)
    opt   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    sch   = ReduceLROnPlateau(opt, patience=3, factor=0.5)
    model = run(model, opt, sch, EPOCHS_HEAD, "Phase-1")

    # Phase 2 — fine-tune last 3 blocks
    print("\n[Phase 2] Fine-tuning last 3 feature blocks ...")
    for layer in list(model.features.children())[-3:]:
        for p in layer.parameters():
            p.requires_grad = True

    opt   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    sch   = ReduceLROnPlateau(opt, patience=3, factor=0.5)
    model = run(model, opt, sch, EPOCHS_FINE, "Phase-2")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx":     train_ds.class_to_idx,
        "img_size":         IMG_SIZE,
    }, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")