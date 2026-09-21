import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PlanDataset
from src.model import UNet

N_CLASSES = 3
CLASS_NAMES = ["background", "wall", "door"]


def dice_loss(logits, target, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = torch.zeros(N_CLASSES)
    union = torch.zeros(N_CLASSES)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        for c in range(N_CLASSES):
            p, t = pred == c, y == c
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
    iou = inter / union.clamp(min=1)
    return iou, iou.mean().item()


def main(epochs=20, bs=16, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    n = len(PlanDataset())
    idx = list(range(n))
    split = int(n * 0.85)
    train_ds = PlanDataset(augment=True, indices=idx[:split])
    val_ds = PlanDataset(augment=False, indices=idx[split:])
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, num_workers=0)

    model = UNet(in_ch=1, n_classes=N_CLASSES, base=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 5.0]).to(device))

    best = 0.0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()

        iou, miou = evaluate(model, val_dl, device)
        per = " ".join(f"{name}={v:.3f}" for name, v in zip(CLASS_NAMES, iou.tolist()))
        print(f"loss={total/len(train_dl):.4f} mIoU={miou:.4f} | {per}")
        if miou > best:
            best = miou
            torch.save(model.state_dict(), "checkpoints/unet_best.pt")
            print("  saved best")

    print("best mIoU:", best)


if __name__ == "__main__":
    main()
