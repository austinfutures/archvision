import sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import UNet
from src.vectorize import mask_to_class_polygons
from src.validate import validate_loop
from src.export_dxf import export_dxf


def load_model(ckpt="checkpoints/unet_best.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=1, n_classes=3, base=8).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, device


@torch.no_grad()
def predict(model, device, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
    return img, model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)


def plot_result(img, pred_mask, walls, doors, out_png):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("input")
    ax[1].imshow(pred_mask, cmap="viridis")
    ax[1].set_title("U-Net mask")
    ax[2].imshow(np.full_like(img, 255), cmap="gray", vmin=0, vmax=255)
    for w in walls:
        x, y = w.exterior.xy
        ax[2].fill(x, y, color="black", alpha=0.7)
        for hole in w.interiors:
            hx, hy = hole.xy
            ax[2].fill(hx, hy, color="white")
    for d in doors:
        x, y = d.exterior.xy
        ax[2].fill(x, y, color="red", alpha=0.7)
    ax[2].set_title("vectorized + validated")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main(img_path):
    model, device = load_model()
    img, pred = predict(model, device, img_path)
    polys = mask_to_class_polygons(pred)
    walls = polys.get(1, [])
    doors = polys.get(2, [])
    print(f"raw: {len(walls)} walls, {len(doors)} doors")

    walls, doors = validate_loop(walls, doors)
    print(f"clean: {len(walls)} walls, {len(doors)} doors")

    export_dxf(walls, doors, "outputs/result.dxf")
    plot_result(img, pred, walls, doors, "outputs/result.png")
    print("wrote outputs/result.dxf and outputs/result.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/images/plan_0799.png"
    main(path)

