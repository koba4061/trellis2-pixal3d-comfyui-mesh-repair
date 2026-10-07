# ======================================================================
#  Mesh Repair Lite — ComfyUI custom node
# ======================================================================
#
#  Author : YosukeKobayashi
#  Date   : 2026-10-07
#
#  Light topology repair for TRELLIS.2 / Pixal3D meshes.
#  Registered in the 3d/mesh category as "Mesh Repair Lite".
#
#  The generated surface is kept. There is no voxel rebuild.
#  Weld, then repair each connected component, then fill holes.
#  Splitting components stops pymeshfix from deleting every part
#  except the largest one. Defects that remain are still exported
#  and counted in the report.
#
#  The repair itself is repair_lite.py. It is reloaded every run,
#  so replacing the file is enough. No ComfyUI restart.
#
#  Requires: trimesh / scipy
#            pymeshfix (preferred) / pymeshlab (fallback)
#            fast-simplification (CPU decimate fallback)
#
# ======================================================================
import importlib
import time

import numpy as np
import torch

try:
    from . import repair_lite as R
except ImportError:
    import repair_lite as R

NODE_ID = "MeshRepairLite"


def _emit(node_id, msg):
    """Show progress on the node and on the console."""
    try:
        from server import PromptServer
        PromptServer.instance.send_progress_text(msg, node_id)
    except Exception:
        pass
    print(f"[MeshRepairLite {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _np(t):
    return None if t is None else t.detach().cpu().numpy()


def _items(mesh):
    """Yield MESH items as (v, f, uv, colors) tensors."""
    v, f = mesh.vertices, mesh.faces
    uv = getattr(mesh, "uvs", None)
    col = getattr(mesh, "vertex_colors", None)
    vc = getattr(mesh, "vertex_counts", None)
    fc = getattr(mesh, "face_counts", None)

    def _at(t, i, n=None):
        if t is None:
            return None
        x = t[i]
        return x[:n] if n is not None else x

    if isinstance(v, list):
        for i in range(len(v)):
            yield v[i], f[i], _at(uv, i), _at(col, i)
    elif torch.is_tensor(v) and v.ndim == 3:
        for i in range(v.shape[0]):
            nv = int(vc[i]) if vc is not None else v.shape[1]
            nf = int(fc[i]) if fc is not None else f.shape[1]
            yield v[i, :nv], f[i, :nf], _at(uv, i, nv), _at(col, i, nv)
    else:
        yield v, f, uv, col


def _pack(vs, fs, uvs, cols, src_mesh):
    """Pack back into Types.MESH. Prefer the Comfy helper."""
    kwargs = {}
    for attr in ("unlit", "occlusion_in_mr", "material", "emissive"):
        kwargs[attr] = getattr(src_mesh, attr, None)
    try:
        from comfy_extras.nodes_save_3d import pack_variable_mesh_batch
        return pack_variable_mesh_batch(vs, fs, colors=cols, uvs=uvs, **kwargs)
    except Exception:
        pass
    try:
        from comfy_api.latest import Types
    except Exception:
        from comfy_api.v0_0_2 import Types
    vv = torch.stack(vs) if len(vs) > 1 else vs[0].unsqueeze(0)
    ff = torch.stack(fs) if len(fs) > 1 else fs[0].unsqueeze(0)
    uu = torch.stack(uvs) if uvs and len(uvs) > 1 else (uvs[0].unsqueeze(0) if uvs else None)
    cc = torch.stack(cols) if cols and len(cols) > 1 else (cols[0].unsqueeze(0) if cols else None)
    try:
        return Types.MESH(vv, ff, uvs=uu, vertex_colors=cc, **kwargs)
    except TypeError:
        return Types.MESH(vv, ff, uvs=uu, vertex_colors=cc)


class MeshRepairLite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "weld_eps": ("FLOAT", {"default": 0.00001, "min": 0.0, "max": 0.01, "step": 0.000001,
                                       "tooltip": "Weld tolerance. Only collapses UV-seam duplicates (~1e-6). Do not raise it."}),
                "input_max_faces": ("INT", {"default": 5000000, "min": 0, "max": 200000000, "step": 100000,
                                            "tooltip": "Face cap for the high-poly mesh used in normal baking. GPU QEM, CPU on failure. 0 = no cut."}),
                "max_faces": ("INT", {"default": 700000, "min": 0, "max": 20000000, "step": 1000,
                                      "tooltip": "Face cap for the final mesh."}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "remove_inner_shell": ("BOOLEAN", {"default": False,
                                                 "tooltip": "Delete a shell fully inside another component. Thin parts can be misread. Aborts if the cut would exceed 30%."}),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MESH", "MESH", "STRING")
    RETURN_NAMES = ("mesh", "mesh_high", "report")
    FUNCTION = "run"
    CATEGORY = "3d/mesh"
    DESCRIPTION = (
        "Light repair toward a watertight manifold mesh. "
        "Weld, decimate, then pymeshfix on each connected component, then fill holes. "
        "Separate parts such as wings are kept. "
        "Requires trimesh. Recommended: pymeshfix, pymeshlab, fast-simplification."
    )

    def run(self, mesh, weld_eps, input_max_faces, max_faces, fill_holes, remove_inner_shell, node_id=None):
        # repair_lite.py is reloaded every run, so a file swap applies
        # without restarting ComfyUI.
        importlib.reload(R)

        emit = lambda m: _emit(node_id, m)
        print(flush=True)

        out_v, out_f, out_uv, out_c = [], [], [], []
        hi_v, hi_f = [], []
        lines = []
        for v, f, uv, col in _items(mesh):
            v_np, f_np = _np(v), _np(f)
            emit(f"Loading input mesh ({len(f_np):,} faces)...")
            if len(f_np) > 8_000_000:
                rep = f"[input] faces={len(f_np)} verts={len(v_np)} (large: stats after weld)"
            else:
                rep = R.report_lines("input", v_np, f_np, R.n_components(f_np))
            lines.append(rep)
            emit(rep)

            ov, of, hv, hf, rep_lines = R.repair_light_topology(
                v_np, f_np,
                weld_eps=weld_eps,
                input_max_faces=input_max_faces,
                max_faces=max_faces,
                fill_holes=fill_holes,
                remove_inner=remove_inner_shell,
                say=emit,
            )
            lines.extend(rep_lines)
            out_v.append(torch.from_numpy(np.asarray(ov, np.float32)))
            out_f.append(torch.from_numpy(np.asarray(of, np.int64)))
            out_uv.append(None)
            col_np = _np(col)
            if col_np is not None and len(col_np) == len(v_np):
                try:
                    from scipy.spatial import cKDTree
                    _, idx = cKDTree(np.asarray(v_np)).query(np.asarray(ov), workers=-1)
                    out_c.append(torch.from_numpy(np.asarray(col_np[idx], np.float32)))
                except Exception:
                    out_c.append(None)
            else:
                out_c.append(None)
            hi_v.append(torch.from_numpy(np.asarray(hv, np.float32)))
            hi_f.append(torch.from_numpy(np.asarray(hf, np.int64)))

        uvs_out = out_uv if all(u is not None for u in out_uv) and out_uv else None
        cols_out = out_c if all(c is not None for c in out_c) and out_c else None
        packed = _pack(out_v, out_f, uvs_out, cols_out, mesh)
        packed_hi = _pack(hi_v, hi_f, None, None, mesh)
        report = "\n".join(lines)
        print(f"[MeshRepairLite {time.strftime('%H:%M:%S')}]\n" + report, flush=True)
        return {"ui": {"text": lines}, "result": (packed, packed_hi, report)}


NODE_CLASS_MAPPINGS = {NODE_ID: MeshRepairLite}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: "Mesh Repair Lite"}
