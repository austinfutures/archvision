import cv2
import numpy as np
from shapely.geometry import Polygon


def mask_to_polygons(binary_mask, min_area=20, simplify_tol=0.5):
    m = (binary_mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    polys = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue
        if cv2.contourArea(cnt) < min_area:
            continue
        shell = cnt.reshape(-1, 2)
        if len(shell) < 3:
            continue
        holes = []
        child = hierarchy[i][2]
        while child != -1:
            h = contours[child].reshape(-1, 2)
            if len(h) >= 3 and cv2.contourArea(contours[child]) >= min_area:
                holes.append(h)
            child = hierarchy[child][0]
        poly = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = poly.simplify(simplify_tol, preserve_topology=True)
        if not poly.is_empty:
            polys.append(poly)
    return polys


def mask_to_class_polygons(class_mask):
    out = {}
    for cid in np.unique(class_mask):
        if cid == 0:
            continue
        out[int(cid)] = mask_to_polygons(class_mask == cid)
    return out
