import ezdxf
from shapely.geometry import Polygon


def _add_poly(msp, poly, layer):
    msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={"layer": layer})
    for hole in poly.interiors:
        msp.add_lwpolyline(list(hole.coords), close=True, dxfattribs={"layer": layer})


def export_dxf(walls, doors, path):
    doc = ezdxf.new("R2010")
    doc.layers.add("WALLS", color=7)
    doc.layers.add("DOORS", color=1)
    msp = doc.modelspace()
    for w in walls:
        if isinstance(w, Polygon):
            _add_poly(msp, w, "WALLS")
    for d in doors:
        if isinstance(d, Polygon):
            _add_poly(msp, d, "DOORS")
    doc.saveas(path)
