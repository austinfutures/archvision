import random
import numpy as np
import cv2
import ezdxf
from shapely.geometry import box
from tqdm import tqdm

IMG_SIZE = 256
WALL_T = 6
DOOR_W = 22


def make_plan(rng):
    margin = rng.randint(20, 40)
    x0, y0, x1, y1 = margin, margin, IMG_SIZE - margin, IMG_SIZE - margin

    walls = [
        box(x0, y0, x1, y0 + WALL_T),
        box(x0, y1 - WALL_T, x1, y1),
        box(x0, y0, x0 + WALL_T, y1),
        box(x1 - WALL_T, y0, x1, y1),
    ]
    doors = []

    for _ in range(rng.randint(2, 4)):
        if rng.random() < 0.5:
            x = rng.randint(x0 + 40, x1 - 40)
            ya, yb = y0 + WALL_T, y1 - WALL_T
            gap_c = rng.randint(ya + DOOR_W, yb - DOOR_W)
            walls.append(box(x, ya, x + WALL_T, gap_c - DOOR_W // 2))
            walls.append(box(x, gap_c + DOOR_W // 2, x + WALL_T, yb))
            doors.append(box(x, gap_c - DOOR_W // 2, x + WALL_T, gap_c + DOOR_W // 2))
        else:
            y = rng.randint(y0 + 40, y1 - 40)
            xa, xb = x0 + WALL_T, x1 - WALL_T
            gap_c = rng.randint(xa + DOOR_W, xb - DOOR_W)
            walls.append(box(xa, y, gap_c - DOOR_W // 2, y + WALL_T))
            walls.append(box(gap_c + DOOR_W // 2, y, xb, y + WALL_T))
            doors.append(box(gap_c - DOOR_W // 2, y, gap_c + DOOR_W // 2, y + WALL_T))

    return walls, doors


def write_dxf(walls, doors, path):
    doc = ezdxf.new("R2010")
    doc.layers.add("WALLS", color=7)
    doc.layers.add("DOORS", color=1)
    msp = doc.modelspace()
    for w in walls:
        msp.add_lwpolyline(list(w.exterior.coords), close=True, dxfattribs={"layer": "WALLS"})
    for d in doors:
        msp.add_lwpolyline(list(d.exterior.coords), close=True, dxfattribs={"layer": "DOORS"})
    doc.saveas(path)


def poly_pts(p):
    return np.array(list(p.exterior.coords), dtype=np.int32).reshape(-1, 1, 2)


def rasterize(walls, doors):
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    img = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)

    for w in walls:
        cv2.fillPoly(mask, [poly_pts(w)], 1)
    for d in doors:
        cv2.fillPoly(mask, [poly_pts(d)], 2)

    for w in walls:
        cv2.polylines(img, [poly_pts(w)], True, 0, 1)
    for d in doors:
        minx, miny, maxx, maxy = [int(v) for v in d.bounds]
        cx, cy = (minx + maxx) // 2, (miny + maxy) // 2
        r = max(maxx - minx, maxy - miny) // 2
        cv2.line(img, (minx, miny), (maxx, maxy), 0, 1)
        cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 90, 0, 1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = (np.random.randn(IMG_SIZE, IMG_SIZE) * 8).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, mask


def main(n=800, seed=0):
    rng = random.Random(seed)
    np.random.seed(seed)
    for i in tqdm(range(n)):
        walls, doors = make_plan(rng)
        write_dxf(walls, doors, f"data/dxf/plan_{i:04d}.dxf")
        img, mask = rasterize(walls, doors)
        cv2.imwrite(f"data/images/plan_{i:04d}.png", img)
        cv2.imwrite(f"data/masks/plan_{i:04d}.png", mask)


if __name__ == "__main__":
    main()
