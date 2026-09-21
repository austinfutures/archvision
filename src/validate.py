from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import explain_validity


def _flatten(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def inspect(walls, doors, min_area=30, door_wall_tol=4.0):
    issues = []
    for i, w in enumerate(walls):
        if not w.is_valid:
            issues.append({"type": "invalid", "idx": i, "detail": explain_validity(w)})
        if w.area < min_area:
            issues.append({"type": "tiny", "idx": i, "detail": f"area={w.area:.1f}"})
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            inter = walls[i].intersection(walls[j])
            if inter.area > 1.0:
                issues.append({"type": "overlap", "idx": (i, j), "detail": f"area={inter.area:.1f}"})
    wall_union = unary_union(walls) if walls else None
    for i, d in enumerate(doors):
        if wall_union is None or d.distance(wall_union) > door_wall_tol:
            issues.append({"type": "floating_door", "idx": i, "detail": "not adjacent to any wall"})
    return issues


def repair(walls, doors, issues, snap_gap=3.0, min_area=30):
    types = {iss["type"] for iss in issues}
    new_walls = list(walls)

    if "invalid" in types:
        new_walls = [w if w.is_valid else w.buffer(0) for w in new_walls]
    if "tiny" in types:
        new_walls = [w for w in new_walls if w.area >= min_area]
    if "overlap" in types:
        merged = unary_union(new_walls)
        new_walls = _flatten(merged)
    if new_walls and snap_gap > 0:
        closed = unary_union([w.buffer(snap_gap / 2) for w in new_walls]).buffer(-snap_gap / 2)
        new_walls = _flatten(closed)

    new_doors = list(doors)
    if "floating_door" in types and new_walls:
        wall_union = unary_union(new_walls)
        new_doors = [d for d in new_doors if d.distance(wall_union) <= 4.0]

    return new_walls, new_doors


def validate_loop(walls, doors, max_iters=5, verbose=True):
    for it in range(max_iters):
        issues = inspect(walls, doors)
        if verbose:
            summary = {}
            for iss in issues:
                summary[iss["type"]] = summary.get(iss["type"], 0) + 1
            print(f"[validate] iter {it}: {len(issues)} issues {summary}")
        if not issues:
            break
        walls, doors = repair(walls, doors, issues)
    return walls, doors
