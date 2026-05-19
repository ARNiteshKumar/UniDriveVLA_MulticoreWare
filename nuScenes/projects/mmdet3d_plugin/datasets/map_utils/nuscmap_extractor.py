"""NuscMapExtractor — extract BEV map geometry from nuScenes HD maps.

Used by nuscenes_converter.py during offline data pre-processing to populate
the 'map_annos' field of each sample info dict.

Returns map elements clipped to an ego-centred ROI, in ego-vehicle BEV frame:
  - 'divider'      : lane dividers and road dividers (LineStrings)
  - 'ped_crossing' : pedestrian crossing polygons as exterior rings (LineStrings)
  - 'boundary'     : road-segment/lane exterior boundaries (LineStrings)

All coordinates are in metres relative to the ego vehicle (x=forward, y=left).
"""

import numpy as np

try:
    from pyquaternion import Quaternion
    from shapely.geometry import LineString, MultiLineString, Polygon, box
    from nuscenes.map_expansion.map_api import NuScenesMap
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

_MAP_NAMES = [
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
]

_DIVIDER_LAYERS = ["lane_divider", "road_divider"]
_CROSSING_LAYERS = ["ped_crossing"]
_BOUNDARY_LAYERS = ["road_segment", "lane"]


def _clip_line(pts_ego, roi):
    """Clip a polyline to an ROI box; return list of LineString segments."""
    if len(pts_ego) < 2:
        return []
    geom = LineString(pts_ego).intersection(roi)
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom] if len(geom.coords) >= 2 else []
    if geom.geom_type == "MultiLineString":
        return [g for g in geom.geoms if len(g.coords) >= 2]
    return []


def _clip_polygon_exterior(pts_ego, roi):
    """Clip a polygon to an ROI; return exterior ring(s) as LineStrings."""
    if len(pts_ego) < 3:
        return []
    try:
        poly = Polygon(pts_ego)
        clipped = poly.intersection(roi)
    except Exception:
        return []
    if clipped.is_empty:
        return []
    polys = (
        [clipped] if clipped.geom_type == "Polygon"
        else list(clipped.geoms) if clipped.geom_type == "MultiPolygon"
        else []
    )
    result = []
    for p in polys:
        coords = list(p.exterior.coords)
        if len(coords) >= 2:
            result.append(LineString(coords))
    return result


class NuscMapExtractor:
    """Extract nuScenes HD-map geometry clipped to an ego-centred BEV ROI.

    Args:
        dataroot: Path to the nuScenes dataset root (contains 'maps/').
        roi_size: (width_m, height_m) of the BEV region of interest, default (30, 60).
    """

    def __init__(self, dataroot, roi_size=(30, 60)):
        self.roi_size = roi_size
        self.maps = {}
        if not _HAS_DEPS:
            return
        for name in _MAP_NAMES:
            try:
                self.maps[name] = NuScenesMap(dataroot=dataroot, map_name=name)
            except Exception:
                pass

    def get_map_geom(self, map_location, translation, rotation):
        """Return map geometry around the ego pose in ego-vehicle BEV coordinates.

        Args:
            map_location: nuScenes map name string (e.g. 'boston-seaport').
            translation:  [x, y, z] ego position in global frame (metres).
            rotation:     quaternion [w, x, y, z] ego heading in global frame.

        Returns:
            dict with keys 'divider', 'ped_crossing', 'boundary',
            each a list of Shapely LineString objects (ego BEV frame, metres).
        """
        result = {"divider": [], "ped_crossing": [], "boundary": []}

        if not _HAS_DEPS or map_location not in self.maps:
            return result

        nusc_map = self.maps[map_location]
        ego_x, ego_y = float(translation[0]), float(translation[1])

        # Ego-frame rotation: global → ego (inverse of ego→global yaw)
        q = Quaternion(rotation)
        yaw = q.yaw_pitch_roll[0]
        cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        def to_ego(pts_global):
            """(N, 2) global xy → ego BEV xy."""
            pts = np.array(pts_global)[:, :2] - [ego_x, ego_y]
            return pts @ R.T

        half_w, half_h = self.roi_size[0] / 2.0, self.roi_size[1] / 2.0
        roi = box(-half_w, -half_h, half_w, half_h)

        # Enlarged global patch for map queries (covers all rotations of the ROI)
        margin = (half_w**2 + half_h**2) ** 0.5 + 2.0
        patch_box = (
            ego_x - margin, ego_y - margin,
            ego_x + margin, ego_y + margin,
        )

        # ── Dividers ───────────────────────────────────────────────────────────
        for layer in _DIVIDER_LAYERS:
            tokens = nusc_map.get_records_in_patch(
                patch_box, [layer], mode="intersect"
            ).get(layer, [])
            for tok in tokens:
                try:
                    rec = nusc_map.get(layer, tok)
                    line_rec = nusc_map.get("line", rec["line_token"])
                    pts = np.array([
                        [nusc_map.get("node", n)["x"],
                         nusc_map.get("node", n)["y"]]
                        for n in line_rec["node_tokens"]
                    ])
                    result["divider"].extend(_clip_line(to_ego(pts), roi))
                except Exception:
                    continue

        # ── Pedestrian crossings ───────────────────────────────────────────────
        tokens = nusc_map.get_records_in_patch(
            patch_box, _CROSSING_LAYERS, mode="intersect"
        ).get("ped_crossing", [])
        for tok in tokens:
            try:
                rec = nusc_map.get("ped_crossing", tok)
                poly = nusc_map.extract_polygon(rec["polygon_token"])
                pts = np.array(poly.exterior.coords)[:, :2]
                result["ped_crossing"].extend(_clip_polygon_exterior(to_ego(pts), roi))
            except Exception:
                continue

        # ── Road boundaries ────────────────────────────────────────────────────
        for layer in _BOUNDARY_LAYERS:
            tokens = nusc_map.get_records_in_patch(
                patch_box, [layer], mode="intersect"
            ).get(layer, [])
            for tok in tokens:
                try:
                    rec = nusc_map.get(layer, tok)
                    poly = nusc_map.extract_polygon(rec["polygon_token"])
                    pts = np.array(poly.exterior.coords)[:, :2]
                    result["boundary"].extend(_clip_polygon_exterior(to_ego(pts), roi))
                except Exception:
                    continue

        return result
