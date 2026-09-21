import random
from shapely.geometry import box
from src.validate import inspect, validate_loop


def summarize(issues):
    s = {}
    for i in issues:
        s[i["type"]] = s.get(i["type"], 0) + 1
    return s


def main():
    random.seed(0)
    # clean wall network: two walls meeting with a hairline gap, plus a duplicate overlap
    walls = [
        box(0, 0, 100, 6),
        box(102.5, 0, 200, 6),      # hairline gap of 2.5px from the first wall
        box(50, 0, 60, 40),         # overlaps the first wall
        box(50, 0, 60, 40),         # exact duplicate of the previous wall
        box(300, 300, 301, 301),    # tiny speck (area 1)
    ]
    doors = [
        box(120, 0, 140, 6),        # door sitting in a wall: fine
        box(500, 500, 520, 506),    # floating door far from every wall
    ]

    print("BEFORE:", summarize(inspect(walls, doors)))
    new_walls, new_doors = validate_loop(walls, doors)
    print("AFTER: ", summarize(inspect(new_walls, new_doors)))
    print(f"walls {len(walls)} -> {len(new_walls)}, doors {len(doors)} -> {len(new_doors)}")


if __name__ == "__main__":
    main()
