"""Build and serve a local blind-review atlas for Step 23A.1.

Run this after the Step 23A.1 ``--phase prepare`` command. The script uses the
already cached 2023 OSM geometries and the official 2023 centreline to create
one local overlay image per frozen review record. It then opens a browser form
whose Save button writes verdicts and notes back to the original review CSV.

The atlas never reads AADT, predictions, errors, residuals, or model decisions.
It automates evidence retrieval and recording; it does not generate a verdict.
"""

from __future__ import annotations

import functools
import html
import importlib.util
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_ROOT / "src" / "23a_test_2023_osm_road_class.py"
REVIEW_CSV = PROJECT_ROOT / "outputs" / "tables" / "step23a1_blind_match_review.csv"
ATLAS_DIR = PROJECT_ROOT / "outputs" / "figures" / "step23a1_blind_review_atlas"
IMAGE_DIR = ATLAS_DIR / "images"
INDEX_PATH = ATLAS_DIR / "index.html"

VERDICTS = (
    "",
    "correct",
    "parallel_carriageway_mismatch",
    "grade_level_mismatch",
    "nonmotorized",
    "wrong_road",
    "indeterminate",
)
FORBIDDEN_REVIEW_COLUMNS = {
    "aadt",
    "observed_aadt",
    "predicted_aadt",
    "prediction",
    "error",
    "residual",
    "absolute_error",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def plot_geometry(axis, geometry, **kwargs) -> None:
    if geometry is None or geometry.is_empty:
        return
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    for part in parts:
        coordinates = np.asarray(part.coords)
        axis.plot(coordinates[:, 0], coordinates[:, 1], **kwargs)


def feature_tags(features: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for feature in features:
        if feature.get("type") != "way" or feature.get("id") is None:
            continue
        osm_id = f"way/{feature['id']}"
        if osm_id in rows:
            continue
        tags = feature.get("tags", {}) or {}
        rows[osm_id] = {
            "highway": first_value(tags.get("highway")),
            "service": first_value(tags.get("service")),
            "name": first_value(tags.get("name:en", tags.get("name", ""))),
            "bridge": first_value(tags.get("bridge")),
            "tunnel": first_value(tags.get("tunnel")),
            "layer": first_value(tags.get("layer")),
            "oneway": first_value(tags.get("oneway")),
        }
    return rows


def tag_summary(tags: dict[str, str]) -> str:
    values = [f"highway={tags.get('highway', '') or 'missing'}"]
    for key in ("service", "bridge", "tunnel", "layer", "oneway"):
        if tags.get(key):
            values.append(f"{key}={tags[key]}")
    return "; ".join(values)


def geometry_inputs():
    base = load_module("hk_aadt_step23a_review_base", BASE_SCRIPT)
    centerline, road_geometries = base.read_centerline(base.find_road_geodatabase())
    print("Reading cached 2023 OSM geometry for the blind-review atlas...")
    features, _ = base.obtain_osm_tiles(centerline)
    ways, osm_geometries, _ = base.parse_osm_ways(features)
    id_to_position = {
        str(osm_id): position for position, osm_id in enumerate(ways["osm_id"])
    }
    return (
        centerline,
        road_geometries,
        osm_geometries,
        id_to_position,
        feature_tags(features),
    )


def draw_record(
    record: pd.Series,
    centerline: pd.DataFrame,
    road_geometries: np.ndarray,
    osm_geometries: np.ndarray,
    id_to_position: dict[str, int],
    tags: dict[str, dict[str, str]],
    tree: STRtree,
) -> Path:
    segment_index = int(record["road_2023_segment_index"])
    official = road_geometries[segment_index]
    corrected_id = str(record.get("osm_id", ""))
    original_id = str(record.get("original_osm_id", ""))
    corrected = (
        osm_geometries[id_to_position[corrected_id]]
        if corrected_id in id_to_position
        else None
    )
    original = (
        osm_geometries[id_to_position[original_id]]
        if original_id in id_to_position
        else None
    )

    figure, axis = plt.subplots(figsize=(9.2, 7.2))
    nearby = tree.query(official.buffer(180.0))
    for position in nearby:
        plot_geometry(
            axis,
            osm_geometries[int(position)],
            color="#B9B9B9",
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )
    if original is not None and original_id != corrected_id:
        plot_geometry(
            axis,
            original,
            color="#F28E2B",
            linewidth=3.0,
            linestyle="--",
            alpha=0.85,
            label="Original Step 23A match",
            zorder=3,
        )
    if corrected is not None:
        plot_geometry(
            axis,
            corrected,
            color="#D62728",
            linewidth=4.2,
            alpha=0.9,
            label="Corrected motor-road OSM match",
            zorder=4,
        )
    plot_geometry(
        axis,
        official,
        color="#1464F4",
        linewidth=5.0,
        alpha=0.95,
        label="Official TD centreline segment",
        zorder=5,
    )
    midpoint = official.interpolate(0.5, normalized=True)
    axis.scatter(
        [midpoint.x],
        [midpoint.y],
        marker="x",
        s=90,
        linewidth=2.3,
        color="#111111",
        label="Official segment midpoint",
        zorder=6,
    )
    minx, miny, maxx, maxy = official.bounds
    extent = max(maxx - minx, maxy - miny, 1.0)
    margin = max(90.0, 0.65 * extent)
    centre_x = (minx + maxx) / 2.0
    centre_y = (miny + maxy) / 2.0
    half = max((maxx - minx) / 2.0, (maxy - miny) / 2.0) + margin
    axis.set_xlim(centre_x - half, centre_x + half)
    axis.set_ylim(centre_y - half, centre_y + half)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)
    axis.set_xlabel("Hong Kong 1980 Grid Easting (m)")
    axis.set_ylabel("Hong Kong 1980 Grid Northing (m)")

    official_row = centerline.iloc[segment_index]
    official_name = str(record.get("official_name", "") or "unnamed")
    corrected_text = tag_summary(tags.get(corrected_id, {}))
    original_text = tag_summary(tags.get(original_id, {}))
    axis.set_title(
        f"{record['audit_id']} | {record['review_stratum']}\n"
        f"Official: {official_name} | elevation={official_row.get('ELEVATION', '')} | "
        f"direction={official_row.get('TRAVEL_DIRECTION', '')}",
        fontsize=12,
    )
    note = f"Corrected {corrected_id}: {corrected_text}"
    if original_id and original_id != corrected_id:
        note += f"\nOriginal {original_id}: {original_text}"
    figure.text(0.02, 0.015, note, fontsize=9, ha="left", va="bottom")
    axis.legend(loc="upper right", fontsize=8.5)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    image_path = IMAGE_DIR / f"{record['audit_id']}.png"
    figure.savefig(image_path, dpi=155, bbox_inches="tight")
    plt.close(figure)
    return image_path


def build_html(review: pd.DataFrame, tags: dict[str, dict[str, str]]) -> None:
    records = []
    for _, row in review.iterrows():
        corrected_id = str(row.get("osm_id", ""))
        original_id = str(row.get("original_osm_id", ""))
        corrected_number = (
            corrected_id.split("/", 1)[1] if corrected_id.startswith("way/") else ""
        )
        original_number = (
            original_id.split("/", 1)[1] if original_id.startswith("way/") else ""
        )
        latitude = float(row["road_latitude"])
        longitude = float(row["road_longitude"])
        records.append(
            {
                "audit_id": str(row["audit_id"]),
                "review_stratum": str(row["review_stratum"]),
                "official_name": str(row.get("official_name", "")),
                "official_alias": str(row.get("official_alias", "")),
                "corrected_id": corrected_id,
                "corrected_tags": tag_summary(tags.get(corrected_id, {})),
                "original_id": original_id,
                "original_tags": tag_summary(tags.get(original_id, {})),
                "distance": row.get("osm_match_distance_m", ""),
                "overlap": row.get("osm_overlap_share", ""),
                "similarity": row.get("osm_name_similarity", ""),
                "image": f"images/{row['audit_id']}.png",
                "google_url": (
                    f"https://www.google.com/maps?q={latitude:.7f},{longitude:.7f}"
                ),
                "osm_url": (
                    f"https://www.openstreetmap.org/way/{corrected_number}"
                    if corrected_number
                    else ""
                ),
                "osm_history_url": (
                    f"https://www.openstreetmap.org/way/{corrected_number}/history"
                    if corrected_number
                    else ""
                ),
                "original_url": (
                    f"https://www.openstreetmap.org/way/{original_number}"
                    if original_number
                    else ""
                ),
                "verdict": str(row.get("reviewer_verdict", "")),
                "note": str(row.get("reviewer_note", "")),
            }
        )
    verdict_options = "".join(
        f'<option value="{html.escape(value)}">'
        f'{html.escape(value or "-- choose --")}</option>'
        for value in VERDICTS
    )
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step 23A.1 blind match review</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f5f7;color:#17202a}}
header{{position:sticky;top:0;background:#17202a;color:white;padding:14px 20px;z-index:5;display:flex;gap:18px;align-items:center}}
header h1{{font-size:18px;margin:0;flex:1}} button{{padding:9px 16px;font-weight:650;cursor:pointer}}
#status{{min-width:190px}} main{{max-width:1250px;margin:18px auto;padding:0 16px}}
.card{{display:grid;grid-template-columns:minmax(520px,1.4fr) minmax(330px,.8fr);gap:18px;background:white;border:1px solid #d7dde3;border-radius:10px;margin:16px 0;padding:15px;box-shadow:0 2px 7px #00000012}}
.card img{{width:100%;border:1px solid #ccd2d8;border-radius:6px}} .meta h2{{margin:0 0 7px;font-size:18px}}
.small{{font-size:13px;color:#4a5560;line-height:1.45}} .links a{{margin-right:10px}}
label{{display:block;font-weight:650;margin-top:13px}} select,textarea{{width:100%;box-sizing:border-box;margin-top:5px;padding:8px;font:inherit}}
textarea{{min-height:84px}} .complete{{border-left:7px solid #2ca25f}} .missing{{border-left:7px solid #d95f0e}}
@media(max-width:900px){{.card{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Step 23A.1 blind geometry review — no AADT or model output</h1><span id="status"></span><button id="save">Save to review CSV</button></header>
<main id="cards"></main>
<script>
const records={payload}; const options={json.dumps(verdict_options)};
function esc(value){{return String(value??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));}}
function links(r){{return [r.google_url?`<a target="_blank" href="${{r.google_url}}">Satellite/map</a>`:'',r.osm_url?`<a target="_blank" href="${{r.osm_url}}">Corrected OSM way</a>`:'',r.osm_history_url?`<a target="_blank" href="${{r.osm_history_url}}">Way history</a>`:'',r.original_url?`<a target="_blank" href="${{r.original_url}}">Original way</a>`:''].join(' ');}}
function render(){{const root=document.getElementById('cards');root.innerHTML='';records.forEach((r,i)=>{{const article=document.createElement('article');article.className='card '+(r.verdict?'complete':'missing');article.innerHTML=`<div><img src="${{r.image}}" alt="geometry overlay"></div><div class="meta"><h2>${{esc(r.audit_id)}} · ${{esc(r.review_stratum)}}</h2><div><b>Official:</b> ${{esc(r.official_name)}} ${{esc(r.official_alias)}}</div><div><b>Corrected:</b> ${{esc(r.corrected_id)}} · ${{esc(r.corrected_tags)}}</div><div><b>Original:</b> ${{esc(r.original_id)}} · ${{esc(r.original_tags)}}</div><p class="small">distance=${{esc(r.distance)}} m · overlap=${{esc(r.overlap)}} · name similarity=${{esc(r.similarity)}}</p><p class="links">${{links(r)}}</p><label>Reviewer verdict<select data-index="${{i}}">${{options}}</select></label><label>Reviewer note<textarea data-note="${{i}}">${{esc(r.note)}}</textarea></label></div>`;root.appendChild(article);article.querySelector('select').value=r.verdict;}});updateStatus();}}
function updateStatus(){{const done=records.filter(r=>r.verdict).length;document.getElementById('status').textContent=`${{done}} / ${{records.length}} adjudicated`;}}
document.addEventListener('change',e=>{{if(e.target.matches('select[data-index]')){{const i=+e.target.dataset.index;records[i].verdict=e.target.value;e.target.closest('.card').className='card '+(e.target.value?'complete':'missing');updateStatus();}}}});
document.addEventListener('input',e=>{{if(e.target.matches('textarea[data-note]'))records[+e.target.dataset.note].note=e.target.value;}});
document.getElementById('save').onclick=async()=>{{const response=await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(records.map(r=>({{audit_id:r.audit_id,reviewer_verdict:r.verdict,reviewer_note:r.note}})))}});const result=await response.json();alert(result.message);updateStatus();}};
render();
</script></body></html>"""
    INDEX_PATH.write_text(page, encoding="utf-8")
    print(f"Saved: {INDEX_PATH.relative_to(PROJECT_ROOT)}")


class ReviewHandler(SimpleHTTPRequestHandler):
    review_csv = REVIEW_CSV

    def log_message(self, format, *args):
        return

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        submitted = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(submitted, list):
            self.send_error(400, "Expected a list")
            return
        review = pd.read_csv(self.review_csv, keep_default_na=False)
        if len(submitted) != len(review):
            self.send_error(400, "Record count changed")
            return
        received = {str(item.get("audit_id", "")): item for item in submitted}
        if set(received) != set(review["audit_id"].astype(str)):
            self.send_error(400, "Audit IDs changed")
            return
        invalid = sorted(
            {
                str(item.get("reviewer_verdict", ""))
                for item in submitted
                if str(item.get("reviewer_verdict", "")) not in VERDICTS
            }
        )
        if invalid:
            self.send_error(400, "Invalid verdict")
            return
        for index, row in review.iterrows():
            item = received[str(row["audit_id"])]
            review.at[index, "reviewer_verdict"] = str(
                item.get("reviewer_verdict", "")
            )
            review.at[index, "reviewer_note"] = str(item.get("reviewer_note", ""))
        review.to_csv(self.review_csv, index=False, encoding="utf-8-sig")
        complete = int((review["reviewer_verdict"] != "").sum())
        response = json.dumps(
            {
                "message": (
                    f"Saved {complete}/{len(review)} verdicts directly to "
                    "outputs/tables/step23a1_blind_match_review.csv"
                )
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def validate_review(review: pd.DataFrame) -> None:
    required = {
        "audit_id",
        "review_stratum",
        "road_2023_segment_index",
        "road_latitude",
        "road_longitude",
        "osm_id",
        "original_osm_id",
        "reviewer_verdict",
        "reviewer_note",
    }
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError("Blind-review CSV lacks: " + ", ".join(missing))
    leaked = sorted(FORBIDDEN_REVIEW_COLUMNS & set(review.columns))
    if leaked:
        raise ValueError(
            "Outcome or model columns leaked into the blind review: "
            + ", ".join(leaked)
        )
    if len(review) != 100 or review["audit_id"].duplicated().any():
        raise ValueError("The frozen blind-review sample must contain 100 unique audit IDs")


def main() -> None:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(
            "Run Step 23A.1 with --phase prepare before building the review atlas"
        )
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    review = pd.read_csv(REVIEW_CSV, keep_default_na=False)
    validate_review(review)
    (
        centerline,
        road_geometries,
        osm_geometries,
        id_to_position,
        tags,
    ) = geometry_inputs()
    tree = STRtree(osm_geometries)
    for position, (_, record) in enumerate(review.iterrows(), start=1):
        draw_record(
            record,
            centerline,
            road_geometries,
            osm_geometries,
            id_to_position,
            tags,
            tree,
        )
        if position % 10 == 0:
            print(f"Rendered {position}/{len(review)} blind-review overlays...")
    build_html(review, tags)

    handler = functools.partial(ReviewHandler, directory=str(ATLAS_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{server.server_port}/index.html"
    print(f"Opening the blind-review interface: {url}")
    print(
        "Use Save to write progress directly to the review CSV; "
        "press Ctrl+C here when finished."
    )
    threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped. Saved verdicts remain in the original CSV.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
