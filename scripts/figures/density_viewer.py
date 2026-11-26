"""Side-by-side electron density of several arms in UglyMol, with the panes' views kept in step.

For each arm given as NAME=DIR (a directory holding phenix's refined_2.mtz and
refined_2.pdb), the map coefficients 2FOFCWT/PH2FOFCWT and FOFCWT/PHFOFCWT (and
ANOM/PHANOM when present) are turned into CCP4 maps with gemmi (sample rate 3,
the whole cell), and an index.html is written with one UglyMol viewer per arm in a
grid. Rotating, zooming or recentring any pane copies its camera and target to all
the others, so the same region is compared everywhere. Contours are in rmsd units
(UglyMol's default, 1.5 sigma for 2mFo-DFc, +-3 sigma for mFo-DFc); keys are
UglyMol's (H for help).

With --truth PDB the true model is loaded in every pane too, drawn in a shifted hue,
so density can be judged against the structure that generated the data.

The page loads its files over HTTP, so serve the output directory:

    uv run python scripts/figures/density_viewer.py --arm laue=DIR --arm p2=DIR ... --out OUT
    uv run python -m http.server 8765 --directory OUT    # then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gemmi

UGLYMOL = "https://cdn.jsdelivr.net/npm/uglymol@0.7.2/uglymol.min.js"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Density comparison</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #111; color: #ddd; font: 13px Helvetica, sans-serif; }}
  #bar {{ padding: 6px 10px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  #grid {{ display: grid; grid-template-columns: repeat({ncols}, 1fr); gap: 4px; height: calc(100% - 40px); padding: 0 4px 4px; box-sizing: border-box; }}
  .pane {{ position: relative; min-height: 0; border: 1px solid #333; }}
  .pane .title {{ position: absolute; top: 4px; left: 6px; z-index: 5; background: rgba(0,0,0,.6); padding: 2px 6px; border-radius: 3px; }}
  .pane .view {{ position: absolute; inset: 0; }}
  .hud {{ position: absolute; bottom: 4px; left: 6px; z-index: 5; font-size: 11px; color: #aaa; }}
  .help {{ display: none; }}
</style>
<script src="{uglymol}"></script>
</head><body>
<div id="bar">
  <b>Density comparison</b>
  <label>map <select id="mapsel">{map_options}</select></label>
  <label><input type="checkbox" id="sync" checked> keep views in step</label>
  <span>drag: rotate; wheel: zoom; shift-drag: pan; ctrl-click: recentre on an atom; H in a pane: UglyMol help</span>
</div>
<div id="grid">{panes}</div>
<script>
const ARMS = {arms_json};
const TRUTH = {truth_json};
const views = [];
ARMS.forEach((a, i) => {{
  const v = new UM.Viewer({{viewer: 'v' + i, hud: 'hud' + i, help: 'help' + i}});
  views.push(v);
  v.load_pdb(a.dir + '/refined_2.pdb', {{}}, () => {{
    if (TRUTH) v.load_pdb(TRUTH, {{hue_shift: 0.5}});
    v.load_maps(a.dir + '/2fofc.ccp4', a.dir + '/fofc.ccp4', {{format: 'ccp4'}});
  }});
}});
// copy the view of the pane being moved to every other pane after each of its renders
let syncing = false;
views.forEach((v, i) => {{
  const render = v.render.bind(v);
  v.render = function () {{
    render();
    if (syncing || !document.getElementById('sync').checked) return;
    syncing = true;
    views.forEach((o, j) => {{
      if (j === i || !o.renderer) return;
      o.camera.position.copy(v.camera.position);
      o.camera.quaternion.copy(v.camera.quaternion);
      o.camera.up.copy(v.camera.up);
      o.camera.zoom = v.camera.zoom;
      o.camera.updateProjectionMatrix();
      o.target.copy(v.target);
      o.redraw_maps();
      o.renderer.render(o.scene, o.camera);
    }});
    syncing = false;
  }};
}});
document.getElementById('mapsel').addEventListener('change', (e) => {{
  const which = e.target.value;
  ARMS.forEach((a, i) => {{
    const v = views[i];
    v.map_bags.forEach((b) => v.clear_el_objects(b));
    v.map_bags.length = 0;
    if (which === 'anom') v.load_map(a.dir + '/anom.ccp4', {{format: 'ccp4', diff_map: true}});
    else v.load_maps(a.dir + '/2fofc.ccp4', a.dir + '/fofc.ccp4', {{format: 'ccp4'}});
  }});
}});
</script>
</body></html>
"""


def write_map(mtz: gemmi.Mtz, f: str, phi: str, path: Path) -> bool:
    if f not in mtz.column_labels() or phi not in mtz.column_labels():
        return False
    grid = mtz.transform_f_phi_to_map(f, phi, sample_rate=3)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(path))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR with refined_2.mtz and refined_2.pdb")
    ap.add_argument("--truth", type=Path, default=None, help="the true model, drawn in every pane")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ncols", type=int, default=0, help="panes per row (default: about square)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arms, has_anom = [], True
    for spec in a.arm:
        name, d = spec.split("=", 1)
        d = Path(d)
        dst = a.out / name
        dst.mkdir(exist_ok=True)
        mtz = gemmi.read_mtz_file(str(d / "refined_2.mtz"))
        ok = write_map(mtz, "2FOFCWT", "PH2FOFCWT", dst / "2fofc.ccp4") and write_map(mtz, "FOFCWT", "PHFOFCWT", dst / "fofc.ccp4")
        if not ok:
            raise SystemExit(f"{name}: refined_2.mtz has no 2FOFCWT/FOFCWT map coefficients")
        has_anom &= write_map(mtz, "ANOM", "PHANOM", dst / "anom.ccp4")
        shutil.copy(d / "refined_2.pdb", dst / "refined_2.pdb")
        arms.append({"name": name, "dir": name})
        print(f"{name}: maps written")
    truth = None
    if a.truth is not None:
        shutil.copy(a.truth, a.out / "truth.pdb")
        truth = "truth.pdb"
    ncols = a.ncols or max(1, round(len(arms) ** 0.5 + 0.49))
    panes = "".join(
        f'<div class="pane"><div class="title">{x["name"]}</div><div class="view" id="v{i}"></div>'
        f'<div class="hud" id="hud{i}"></div><div class="help" id="help{i}"></div></div>'
        for i, x in enumerate(arms))
    opts = '<option value="2fofc">2mFo-DFc and mFo-DFc</option>' + (
        '<option value="anom">anomalous difference</option>' if has_anom else "")
    (a.out / "index.html").write_text(PAGE.format(ncols=ncols, uglymol=UGLYMOL, panes=panes, map_options=opts,
                                                  arms_json=json.dumps(arms), truth_json=json.dumps(truth)))
    print(f"wrote {a.out / 'index.html'}; serve with: uv run python -m http.server 8765 --directory {a.out}")


if __name__ == "__main__":
    main()
