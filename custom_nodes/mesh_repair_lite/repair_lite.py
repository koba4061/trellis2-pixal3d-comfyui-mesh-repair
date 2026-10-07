# ======================================================================
#  Mesh Repair Lite — light topology repair for TRELLIS.2 / Pixal3D
# ======================================================================
#
#  Author : YosukeKobayashi
#  Date   : 2026-10-07
#  Target : ComfyUI custom node (Google Colab)
#
#  Summary:
#    A TRELLIS.2 / Pixal3D mesh looks clean and is still a triangle soup:
#    vertices are not welded.
#    Measured: 99.8% of boundary=269,585 are near-duplicate vertices
#    (UV seams / export splits).
#    Real defects are about boundary ~550 / nonmanifold ~750.
#
#    No voxel closing. A light chain keeps the surface:
#    weld, per-component repair, fill small holes,
#    toward watertight manifold geometry.
#
#    Place it after RemeshMesh(udf) and DecimateMesh.
#    A raw O-Voxel mesh also runs through the same path.
#
#  Pipeline:
#      weld(1e-5) -> decimate (GPU QEM / cluster+QEM / CPU)
#      -> split components -> pymeshfix.repair() each one
#        (pymeshlab if that component fails) -> join
#      -> optional inner shell -> unify normals -> report
#
#    Split first because pymeshfix.repair() on many components
#    can delete everything but the largest (measured 27 -> 1).
#    repair() on one component does not delete the others.
#    That is required to keep wings, hinges, and trim.
#
#    Measured: 680k faces, parts=21, boundary/nonmanifold
#    339/3501 -> 0/0, watertight=True, parts kept.
#    No self-intersection removal. It misreads hollow walls
#    and deletes faces. Internal intersections are not visible.
#
#  Policy:
#    - Still export if defects remain, and report the counts.
#      Look is preferred over a destroyed surface.
#    - Dependencies are pip packages only:
#      trimesh / scipy / numpy / pymeshfix (main) / pymeshlab (fallback)
#      fast-simplification (CPU decimate)
#      On ComfyUI, GPU QEM from comfy_extras is used when present.
#
# ======================================================================
from __future__ import annotations

import time
import threading
import numpy as np
import trimesh


# ======================================================================
#  Defaults
# ======================================================================
WELD_EPS = 1e-5                  # weld tolerance; only UV-seam duplicates
DECIMATE_FACES = 700_000         # final face target
PRE_DECIMATE_FACES = 5_000_000   # high-poly cap for repair and normal baking
QEM_DIRECT_MAX = 60_000_000      # at or below this, run GPU QEM directly (40GB VRAM)
CLUSTER_TARGET_FACES = 45_000_000  # cluster pre-pass target so QEM can run
INNER_SHELL_KEEP_RATIO = 0.5     # face ratio of the largest part always kept


# ======================================================================
#  Utilities
# ======================================================================
class _Beat:
    """Heartbeat for long calls. Logs elapsed time every interval seconds.

    Usage:
        with _Beat("label", say):
            heavy_call()
    """
    def __init__(self, label, say=None, interval=15.0):
        self._label = label
        self._say = say if callable(say) else (lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}", flush=True))
        self._interval = float(interval)
        self._stop = threading.Event()

    def __enter__(self):
        t0 = time.time()
        self._say(f"{self._label} ...")
        stop, say, label, iv = self._stop, self._say, self._label, self._interval

        def _loop():
            while not stop.wait(iv):
                say(f"... {label} still running ({time.time() - t0:.0f}s)")

        threading.Thread(target=_loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False


def to_trimesh(v, f):
    return trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(f, np.int64), process=False)


def edge_stats_faces(f):
    """From faces (N,3): (boundary edge count, non-manifold edge count)"""
    f = np.asarray(f)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum())


def n_components(f):
    """Connected components from faces (shared vertices, referenced verts only).
    Returns -1 on failure. Use parts= to see if wings or doors survived."""
    try:
        from scipy import sparse
        used, inv = np.unique(f, return_inverse=True)
        f2 = inv.reshape(-1, 3).astype(np.int32)
        e = np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]])
        a = sparse.csr_matrix(
            (np.ones(len(e), np.int8), (e[:, 0], e[:, 1])),
            shape=(len(used), len(used)))
        return int(sparse.csgraph.connected_components(a, directed=False)[0])
    except Exception:
        return -1


def split_components(v, f):
    """Split (v, f) into [(v_i, f_i), ...] per connected component.
    A face's component comes from its vertex labels. Vertices are
    remapped to local indices. On failure or one part, return [(v, f)]."""
    try:
        from scipy import sparse
        used, inv = np.unique(f, return_inverse=True)
        f2 = inv.reshape(-1, 3).astype(np.int32)
        e = np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]])
        a = sparse.csr_matrix(
            (np.ones(len(e), np.int8), (e[:, 0], e[:, 1])),
            shape=(len(used), len(used)))
        n, lab = sparse.csgraph.connected_components(a, directed=False)
        if n <= 1:
            return [(np.asarray(v), np.asarray(f))]
        uv = np.asarray(v)[used]
        out = []
        for k in range(n):
            fk = np.asarray(f)[lab[f2[:, 0]] == k]
            u2, inv2 = np.unique(fk, return_inverse=True)
            out.append((uv[u2], inv2.reshape(-1, 3)))
        return out
    except Exception:
        return [(np.asarray(v), np.asarray(f))]


def concat_meshes(pairs):
    """Join [(v, f), ...] into one mesh. Face indices are offset."""
    vs, fs, off = [], [], 0
    for vv, ff in pairs:
        vs.append(np.asarray(vv))
        fs.append(np.asarray(ff) + off)
        off += len(vv)
    return np.concatenate(vs), np.concatenate(fs)


def report_lines(label, v, f, parts=None):
    b, nm = edge_stats_faces(f)
    s = (f"[{label}] faces={len(f)} verts={len(v)} "
         f"boundary={b} nonmanifold={nm}")
    if parts is not None:
        s += f" parts={parts}"
    return s


# ======================================================================
#  1) Weld — turn a triangle soup into a mesh. Look is unchanged.
# ======================================================================
def weld_vertices(v, f, eps=WELD_EPS):
    """Weld vertices within eps by rounding. Drop degenerate faces."""
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    q = np.round(v / float(eps)).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    f2 = inv[f]
    keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    f2 = f2[keep]
    # Average the representatives so welded positions stay stable
    nv = int(inv.max()) + 1
    acc = np.zeros((nv, 3), np.float64)
    cnt = np.zeros(nv, np.float64)
    np.add.at(acc, inv, v)
    np.add.at(cnt, inv, 1.0)
    v2 = acc / np.maximum(cnt, 1.0)[:, None]
    return v2, f2


# ======================================================================
#  2) Decimate — GPU QEM first, CPU on failure
# ======================================================================
def _gpu_decimate(v, f, max_faces, say):
    """Run ComfyUI QEM on the GPU.

    Returns (v, f, done). done=True means the mesh is at or under max_faces.
    None means the GPU path failed and the caller should decimate on CPU
    from the original mesh. done=False is a cluster intermediate the CPU can finish.

    free_memory() runs first, then the fastest route for this size:
    pick one:
      <= QEM_DIRECT_MAX   : QEM directly (best quality)
      >  QEM_DIRECT_MAX   : light cluster cut (~CLUSTER_TARGET_FACES,
                           under about 2x, so the grid error stays small) then QEM
    """
    try:
        import torch
        import comfy.model_management as mm
        from comfy_extras.mesh3d.postprocess.qem_decimate import (
            QEMConfig, qem_decimate_simplify, qem_cluster_decimate)
    except Exception:
        say("GPU decimate unavailable (import failed) -> CPU")
        return None
    v_t = f_t = None
    try:
        if not torch.cuda.is_available():
            return None
        dev = mm.get_torch_device()
        with _Beat("free model VRAM", say):
            mm.free_memory(int(len(f)) * 1536 + mm.minimum_inference_memory(), dev)
        with _Beat("upload mesh to GPU", say):
            v_t = torch.from_numpy(np.ascontiguousarray(v, np.float32)).to(dev)
            f_t = torch.from_numpy(np.ascontiguousarray(f, np.int64)).to(dev)

        def _finish(vv_t, ff_t):
            vv = vv_t.detach().cpu().numpy()
            ff = ff_t.detach().cpu().numpy()
            if len(ff) == 0:
                raise RuntimeError("GPU decimate returned an empty mesh")
            return np.asarray(vv, np.float64), np.asarray(ff, np.int64)

        def _qem(vv_t, ff_t):
            with _Beat(f"GPU QEM {ff_t.shape[0]} → {max_faces}", say):
                return qem_decimate_simplify(
                    vv_t, ff_t, int(max_faces), config=QEMConfig())[:2]

        if len(f) <= QEM_DIRECT_MAX:
            try:
                return (*_finish(*_qem(v_t, f_t)), True)
            except Exception as e:
                say(f"GPU QEM failed ({type(e).__name__}: {str(e)[:100]}) -> cluster path")
                torch.cuda.empty_cache()

        # Huge input, or QEM OOM: cluster down to a size QEM can take
        # Target is about CLUSTER_TARGET_FACES, a coarse cut under 2x
        try:
            with _Beat(f"GPU cluster {f_t.shape[0]} faces", say):
                # Thin meshes are about 0.5 verts/face. Ask for a few extra verts.
                v_t, f_t, _ = qem_cluster_decimate(
                    v_t, f_t,
                    target_verts=max(int(CLUSTER_TARGET_FACES * 0.6), 1_000_000))
            say(f"  -> {f_t.shape[0]} faces")
        except Exception as e:
            say(f"GPU cluster failed ({type(e).__name__}: {str(e)[:100]}) -> CPU")
            return None
        if f_t.shape[0] <= max_faces:
            return (*_finish(v_t, f_t), True)
        try:
            v_t, f_t = _qem(v_t, f_t)
        except Exception as e:
            # Hand the clustered intermediate to the CPU fallback
            say(f"GPU QEM failed ({type(e).__name__}) -> intermediate mesh to CPU")
            return (*_finish(v_t, f_t), False)
        return (*_finish(v_t, f_t), True)
    except Exception as e:
        say(f"GPU decimate failed ({type(e).__name__}: {str(e)[:120]}) -> CPU")
        return None
    finally:
        try:
            del v_t, f_t
            torch.cuda.empty_cache()
        except Exception:
            pass


def _cpu_decimate(v, f, max_faces, say):
    """Decimate with fast_simplification (QEM-class, close to GPU QEM)."""
    try:
        import fast_simplification
        red = 1.0 - float(max_faces) / len(f)
        with _Beat(f"CPU decimate {len(f)} → {max_faces}", say):
            vv, ff = fast_simplification.simplify(
                np.asarray(v, np.float32), np.asarray(f, np.int64),
                target_reduction=red)
        if len(ff) == 0:
            raise RuntimeError("CPU decimate returned an empty mesh")
        return np.asarray(vv, np.float64), np.asarray(ff, np.int64)
    except Exception as e:
        say(f"CPU decimate failed ({type(e).__name__}: {str(e)[:120]}) -> keep input")
        return np.asarray(v, np.float64), np.asarray(f, np.int64)


def decimate(v, f, max_faces, say=None):
    """Cut to max_faces or below. Any size. GPU first.

    Route by input face count:
      <= QEM_DIRECT_MAX  -> GPU QEM directly
      >  QEM_DIRECT_MAX  -> light GPU cluster, then GPU QEM
      GPU fully failed   -> CPU fast_simplification
      cluster only       -> finish that intermediate on CPU (faster)
    """
    _say = say if callable(say) else (lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}", flush=True))
    if not max_faces or len(f) <= max_faces:
        return np.asarray(v, np.float64), np.asarray(f, np.int64)
    if len(f) > QEM_DIRECT_MAX:
        _say(f"input {len(f)} faces > {QEM_DIRECT_MAX}: cluster pre-pass + GPU QEM")
    out = _gpu_decimate(v, f, max_faces, _say)
    print()  # clear a leftover tqdm line so the next log starts at column 0
    if out is None:
        return _cpu_decimate(v, f, max_faces, _say)
    vv, ff, done = out
    if not done:
        # Finish the clustered intermediate on CPU. Much faster than the raw mesh.
        return _cpu_decimate(vv, ff, max_faces, _say)
    return vv, ff


# ======================================================================
#  3) Light repair — pymeshfix does non-manifold edges and holes together
# ======================================================================
def pymeshfix_repair(v, f, say=None):
    """Repair non-manifold edges, holes, and degenerates with pymeshfix.

    Measured (Pixal3D, 699k faces): boundary=552 nonmanifold=748 -> 0/0 watertight.
    The caller should pass one component at a time. Several components
    can lose every part but the largest (measured 27 -> 1).
    remove_smallest_components=False is also passed, as a backup.
    No self-intersection removal. It misreads hollow inner and outer
    walls and deletes faces. Returns None on failure.
    """
    _say = say if callable(say) else (lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}", flush=True))
    try:
        import pymeshfix
    except Exception as e:
        _say(f"pymeshfix not installed ({type(e).__name__}) -> pymeshlab")
        return None
    try:
        mf = pymeshfix.MeshFix(np.asarray(v, np.float64), np.asarray(f, np.int32))
        with _Beat("pymeshfix repair (non-manifold + holes)", _say):
            # The default remove_smallest_components=True deletes every
            # component but the largest, so a door or wing on a hinge
            # disappears. Leave component deletion to the inner-shell step.
            try:
                mf.repair(remove_smallest_components=False)
            except TypeError:
                mf.repair()
        vv = np.asarray(mf.points, np.float64)
        ff = np.asarray(mf.faces, np.int64)
        if vv.size == 0 or ff.size == 0:
            raise RuntimeError("pymeshfix returned an empty mesh")
        if len(ff) < max(10, int(len(f) * 0.05)):
            # Detect a collapse that deletes almost every fragment, and
            # fall back. A ratio avoids flagging a small component that is fine.
            raise RuntimeError(f"pymeshfix collapsed ({len(f)}->{len(ff)} faces)")
        n_in, n_out = n_components(f), n_components(ff)
        if n_in >= 0 and n_out >= 0 and n_out < n_in:
            _say(f"  warning: components {n_in}->{n_out} (a part may have been removed)")
        return vv, ff
    except Exception as e:
        _say(f"pymeshfix failed ({type(e).__name__}: {str(e)[:100]}) -> pymeshlab")
        return None


def pymeshlab_repair(v, f, close_holes=True, fill_max_size=0, say=None):
    """Light repair with PyMeshLab filters. Returns (v, f, notes).

    Order: drop duplicate/degenerate faces, snap borders (close tiny cracks),
          repair non-manifold edges, optionally fill holes.
    """
    _say = say if callable(say) else (lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}", flush=True))
    notes = []
    try:
        import pymeshlab
    except Exception as e:
        _say(f"pymeshlab not installed ({type(e).__name__}) -> trimesh only")
        return np.asarray(v), np.asarray(f), ["pymeshlab missing"]

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(v, np.float64),
                               face_matrix=np.asarray(f, np.int64)))
    steps = [
        ("remove duplicate faces", "meshing_remove_duplicate_faces", {}),
        ("remove null faces", "meshing_remove_null_faces", {}),
        ("snap borders (close tiny cracks)", "meshing_snap_mismatched_borders", {}),
        # Split Vertices splits verts and keeps faces. Remove Faces adds boundary.
        ("repair non-manifold edges", "meshing_repair_non_manifold_edges",
         {"method": "Split Vertices"}),
        ("remove unreferenced vertices", "meshing_remove_unreferenced_vertices", {}),
    ]
    for label, filt, kw in steps:
        fn = getattr(ms, filt, None)
        if fn is None:
            notes.append(f"{filt} missing")
            continue
        try:
            with _Beat(f"pymeshlab {label}", _say):
                fn(**kw)
        except Exception as e:
            notes.append(f"{filt} failed: {type(e).__name__}")
    if close_holes:
        try:
            kw = {}
            if fill_max_size and fill_max_size > 0:
                kw["maxholesize"] = int(fill_max_size)
            with _Beat("pymeshlab fill holes", _say):
                ms.meshing_close_holes(**kw)
        except Exception as e:
            notes.append(f"close_holes failed: {type(e).__name__}")

    m = ms.current_mesh()
    vv = np.asarray(m.vertex_matrix(), np.float64)
    ff = np.asarray(m.face_matrix(), np.int64)
    return vv, ff, notes


def trimesh_fill_holes(v, f, say=None):
    """Hole fill when pymeshlab is absent. Closes every boundary loop."""
    _say = say if callable(say) else (lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}", flush=True))
    mesh = to_trimesh(v, f)
    try:
        with _Beat("trimesh fill holes", _say):
            n_filled = int(mesh.fill_holes())
        _say(f"  fill_holes: filled {n_filled} holes")
    except Exception as e:
        _say(f"fill_holes failed: {type(e).__name__}")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


# ======================================================================
#  4) Optional inner shell — drop a component fully inside another
# ======================================================================
def remove_inner_shells(v, f, say=None):
    """Delete components fully inside another. Returns (v, f, removed_faces)"""
    _say = say if callable(say) else (lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}", flush=True))
    mesh = to_trimesh(v, f)
    try:
        with _Beat("split components", _say):
            parts = mesh.split(only_watertight=False)
        if len(parts) <= 1:
            return np.asarray(v), np.asarray(f), 0
        # Always keep the largest. Delete another only if it is inside the rest.
        parts = sorted(parts, key=lambda p: -len(p.faces))
        keep = [parts[0]]
        removed = 0
        for p in parts[1:]:
            others = trimesh.util.concatenate(keep)
            inside = False
            try:
                # Is a sample of this part inside the others? Odd-even ray test.
                pts = p.vertices[:: max(1, len(p.vertices) // 32)]
                inside = bool(others.contains(pts).all())
            except Exception:
                inside = False
            if inside:
                removed += len(p.faces)
            else:
                keep.append(p)
        if removed:
            if removed > len(f) * 0.3:
                _say(f"  inner shell skipped: planned cut {removed} faces ({removed / len(f):.0%}) is too large -> keep all")
                return np.asarray(v), np.asarray(f), 0
            out = trimesh.util.concatenate(keep)
            _say(f"  inner shell: removed {removed} faces")
            return np.asarray(out.vertices), np.asarray(out.faces), removed
        return np.asarray(v), np.asarray(f), 0
    except Exception as e:
        _say(f"inner shell skipped: {type(e).__name__}: {str(e)[:100]}")
        return np.asarray(v), np.asarray(f), 0


# ======================================================================
#  Entry point
# ======================================================================
def repair_light_topology(v, f,
                          weld_eps=WELD_EPS,
                          input_max_faces=PRE_DECIMATE_FACES,
                          max_faces=DECIMATE_FACES,
                          fill_holes=True,
                          remove_inner=False,
                          say=None):
    """Light topology repair -> (out_v, out_f, hi_v, hi_f, report_lines)

    out_*: repaired mesh at or under max_faces
    hi_* : high poly for normal baking (weld + decimate only, before repair)
    Defects that remain are still exported and counted in the report.
    """
    _say = say if callable(say) else (lambda m: print(f"  [{time.strftime('%H:%M:%S')}] {m}", flush=True))
    lines = []
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)

    # -- 1) Weld: triangle soup -> mesh. Most important. Look unchanged.
    with _Beat(f"weld vertices eps={weld_eps}", _say):
        v, f = weld_vertices(v, f, weld_eps)
    parts_after_weld = n_components(f)
    lines.append(report_lines("welded", v, f, parts_after_weld))
    _say(lines[-1])

    # -- 2) Keep a high-poly mesh for baking, then cut to the final count
    v_hi, f_hi = decimate(v, f, input_max_faces, _say)
    lines.append(report_lines("hi", v_hi, f_hi))

    # Cut the final mesh from the high-poly one. Do not decimate the raw mesh twice.
    if max_faces and len(f_hi) > max_faces:
        v, f = decimate(v_hi, f_hi, max_faces, _say)
        # Weld again in case the cut opened degenerates
        v, f = weld_vertices(v, f, weld_eps)
    else:
        v, f = v_hi, f_hi
    lines.append(report_lines("decimated", v, f, n_components(f)))
    _say(lines[-1])

    # -- 3) Per-component repair so pymeshfix cannot delete parts, then pymeshlab, then trimesh
    # On some builds remove_smallest_components=False is ignored and
    # every part but the largest is deleted (measured 27 -> 1). Split first.
    comp = split_components(v, f)
    if len(comp) > 1:
        _say(f"repairing {len(comp)} components separately (keeps parts)")
        fixed = []
        for i, (pv, pf) in enumerate(comp):
            r = pymeshfix_repair(pv, pf, say=_say)
            if r is None:
                rv, rf, notes = pymeshlab_repair(pv, pf, close_holes=fill_holes, say=_say)
                for n in notes:
                    lines.append(f"  note(component {i + 1}): {n}")
                fixed.append((rv, rf))
            else:
                fixed.append(r)
        v, f = concat_meshes(fixed)
    else:
        out = pymeshfix_repair(v, f, say=_say)
        if out is not None:
            v, f = out
        else:
            v, f, notes = pymeshlab_repair(v, f, close_holes=fill_holes, say=_say)
            for n in notes:
                lines.append(f"  note: {n}")
            if fill_holes and "pymeshlab missing" in notes:
                v, f = trimesh_fill_holes(v, f, _say)

    # -- 4) Optional inner shell
    if remove_inner:
        v, f, _rm = remove_inner_shells(v, f, _say)

    # -- 5) Unify normals. Look only. Topology unchanged.
    try:
        m = to_trimesh(v, f)
        trimesh.repair.fix_normals(m)
        v, f = np.asarray(m.vertices), np.asarray(m.faces)
    except Exception:
        pass

    # -- 6) Final check. Remaining defects are reported, not dropped.
    b_end, nm_end = edge_stats_faces(f)
    mesh = to_trimesh(v, f)
    try:
        watertight = bool(mesh.is_watertight)
    except Exception:
        watertight = b_end == 0
    parts_end = n_components(f)
    lines.append(report_lines("repaired", v, f, parts_end))
    lines.append(f"[repaired] watertight={watertight}")
    if parts_after_weld >= 0 and parts_end >= 0 and parts_end < parts_after_weld:
        lines.append(f"  warning: components {parts_after_weld}->{parts_end}"
                     " (a part may have been removed)")
    if b_end > 0 or nm_end > 0:
        lines.append(f"  unresolved: boundary={b_end} nonmanifold={nm_end}"
                     " (exported as-is; look is preferred)")
    for l in lines[-3:] if (b_end or nm_end) else lines[-2:]:
        _say(l)
    return v, f, v_hi, f_hi, lines
