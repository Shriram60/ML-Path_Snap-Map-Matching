"""Folium map of one trip: GPS points, nearby roads, predicted route and,
for development trips only, the reference route.

Run from the repo root:
    python -m visualisations.route_map --trip 00000033 --partition train
    python -m visualisations.route_map --all --partition student_test

Maps are saved to visualisations/output/<partition>_<trip>.html.
The prediction comes from the same Pipeline class the API uses, so the map
shows exactly what the demo would show.
"""
import argparse
import logging
from pathlib import Path

import folium
import numpy as np
from branca.element import Element
from scipy.spatial import cKDTree

from api import config
from api.pipeline import Pipeline

OUT_DIR = Path(__file__).resolve().parent / "output"
DEV_PARTITIONS = ("train", "student_test")  # reference route may be drawn only for these
ROAD_RADIUS_M = 150     # draw roads with an end node within this distance of the track
MAX_GPS_DRAWN = 2000    # thin out long trips so the HTML stays light
EARTH_R = 6371000.0

COLOURS = {"gps": "#2b59c3", "unmatched": "#f79009", "road": "#98a2b3",
           "predicted": "#d92d20", "reference": "#12b76a"}


def to_metres(lonlat, lon0, lat0):
    x = EARTH_R * np.radians(lonlat[:, 0] - lon0) * np.cos(np.radians(lat0))
    y = EARTH_R * np.radians(lonlat[:, 1] - lat0)
    return np.column_stack([x, y])


def nearby_arcs(track, nodes, arcs):
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    tree = cKDTree(to_metres(track[:, :2], lon0, lat0))
    dist, _ = tree.query(to_metres(nodes, lon0, lat0), distance_upper_bound=ROAD_RADIUS_M)
    near_node = np.isfinite(dist)
    return np.where(near_node[arcs[:, 0]] | near_node[arcs[:, 1]])[0]


def arc_lines(arc_ids, nodes, arcs):
    return [[[nodes[a, 1], nodes[a, 0]], [nodes[b, 1], nodes[b, 0]]] for a, b in arcs[arc_ids]]


def legend_html(show_reference):
    rows = [("gps", "GPS point (matched)", "dot"), ("unmatched", "GPS point (no road assigned)", "dot"),
            ("road", "Nearby road network", "line"), ("predicted", "Predicted route", "line")]
    if show_reference:
        rows.append(("reference", "Reference route (dev only)", "dash"))
    items = ""
    for key, text, kind in rows:
        if kind == "dot":
            sw = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{COLOURS[key]}"></span>'
        else:
            border = "dashed" if kind == "dash" else "solid"
            sw = f'<span style="display:inline-block;width:20px;border-top:4px {border} {COLOURS[key]};vertical-align:middle"></span>'
        items += f"<div>{sw}&nbsp; {text}</div>"
    items += ('<div><span style="color:#12b76a">&#9679;</span> start &nbsp; '
              '<span style="color:#101828">&#9679;</span> end</div>')
    return (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;'
            f'border:1px solid #999;border-radius:6px;font:12px sans-serif;line-height:1.8">{items}</div>')


def build_map(pipeline, trip_dir, partition, show_reference=True):
    trip = pipeline.funcs["load_trip"](trip_dir)
    result = pipeline.predict(trip_dir, log_prefix="map")

    track = trip.track[["lon", "lat"]].to_numpy(dtype=float)
    nodes = np.asarray(trip.nodes, dtype=float)
    arcs = np.asarray(trip.arcs, dtype=np.int64)
    show_reference = show_reference and partition in DEV_PARTITIONS and trip.route is not None

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[track[:, 1].min(), track[:, 0].min()], [track[:, 1].max(), track[:, 0].max()]], padding=(30, 30))

    roads = folium.FeatureGroup(name="Nearby road network")
    road_lines = arc_lines(nearby_arcs(track, nodes, arcs), nodes, arcs)
    if road_lines:  # one multi-line object keeps the HTML small
        folium.PolyLine(road_lines, color=COLOURS["road"], weight=2, opacity=0.6).add_to(roads)
    roads.add_to(m)

    if show_reference:
        ref = folium.FeatureGroup(name="Reference route (dev only)")
        folium.PolyLine(arc_lines(np.asarray(trip.route, dtype=np.int64), nodes, arcs),
                        color=COLOURS["reference"], weight=6, opacity=0.7, dash_array="8 6").add_to(ref)
        ref.add_to(m)

    pred = folium.FeatureGroup(name="Predicted route")
    if len(result["route"]) > 1:
        folium.PolyLine(result["route"], color=COLOURS["predicted"], weight=3.5, opacity=0.9).add_to(pred)
    pred.add_to(m)

    step = max(1, len(result["gps"]) // MAX_GPS_DRAWN)
    gps = folium.FeatureGroup(name=f"GPS points (every {step})" if step > 1 else "GPS points")
    for i in range(0, len(result["gps"]), step):  # small circles, coloured by whether a road was assigned
        arc = result["predicted"][i]["arc_id"]  # predicted[i] belongs to gps[i]
        folium.CircleMarker(result["gps"][i], radius=2.5, weight=0, fill=True, fill_opacity=0.8,
                            fill_color=COLOURS["gps"] if arc >= 0 else COLOURS["unmatched"],
                            tooltip=f"point {result['predicted'][i]['obs_idx']}, arc {arc if arc >= 0 else 'none'}").add_to(gps)
    gps.add_to(m)

    ends = folium.FeatureGroup(name="Start / end")
    folium.CircleMarker(result["gps"][0], radius=7, color="white", fill=True, fill_color="#12b76a",
                        fill_opacity=1, tooltip="start").add_to(ends)
    folium.CircleMarker(result["gps"][-1], radius=7, color="white", fill=True, fill_color="#101828",
                        fill_opacity=1, tooltip="end").add_to(ends)
    ends.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(Element(legend_html(show_reference)))
    met = result["metrics"]
    title = (f"{partition} trip {result['trip_id']} | top-1 {met['top1_accuracy']} | "
             f"continuity {met['route_continuity']}")
    if pipeline.uses_placeholder:
        title = "PLACEHOLDER PIPELINE, NOT FOR THE REPORT | " + title
    m.get_root().html.add_child(Element(
        f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px 10px;'
        f'border:1px solid #999;border-radius:6px;font:13px sans-serif">{title}</div>'))
    return m


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", choices=["train", "student_test"], default="student_test")
    ap.add_argument("--trip", help="trip id, for example 00000033")
    ap.add_argument("--all", action="store_true", help="map every trip in the partition")
    ap.add_argument("--no-reference", action="store_true", help="hide the reference route")
    args = ap.parse_args()

    base = config.data_dir()
    if base is None:
        ap.error("set PATHSNAP_DATA first")
    folder = base / args.partition
    if args.all:
        trips = sorted(p.name for p in folder.iterdir() if p.is_dir())
    elif args.trip:
        trips = [args.trip]
    else:
        ap.error("give --trip or --all")

    pipeline = Pipeline()
    OUT_DIR.mkdir(exist_ok=True)
    for tid in trips:
        m = build_map(pipeline, folder / tid, args.partition, show_reference=not args.no_reference)
        out = OUT_DIR / f"{args.partition}_{tid}.html"
        m.save(out)
        print(f"saved {out.relative_to(config.REPO_ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
