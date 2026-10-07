# TRELLIS.2 / Pixal3D — Colab mesh repair (V2)

Yosuke Kobayashi（小林洋介）  
https://yosuke4061.com/

https://yosuke4061.com/new_toppage/briefings/brief155/  
https://yosuke4061.com/new_toppage/briefings/brief155_en/

TRELLIS.2 and Pixal3D turn one image into a 3D mesh in ComfyUI 0.38.2. The mesh looks clean and still has holes and non-manifold edges. A double shell often stays, because UDF remeshing builds a thin wall from both sides of a sheet.

Mesh Repair Lite welds the split vertices, then repairs each connected part. Boundary edges and non-manifold edges go to zero. A closed inner shell can remain. Watertight does not mean the double shell is gone.

Run the notebook in Colab from top to bottom. Use an A100 GPU runtime. Do not change the runtime after you start. The notebook clones this repo for the repair node. No Google Drive, no ngrok, no SSH tunnel. The last cell prints a URL. Open it in the same Google account that has the notebook open.

TRELLIS.2 と Pixal3D は、1 枚の画像から 3D メッシュを作ります。見た目はきれいでも、穴と非多様体が残ります。UDF は薄い面の両側から殻を作るので、二重シェルも残りやすいです。

Mesh Repair Lite は、割れた頂点を溶接してから、部品ごとに直します。境界と非多様体は 0 まで落ちます。閉じた内殻は、水密のまま残ることがあります。

ノートは Colab で上から実行します。GPU は A100 です。始めたあとにランタイムは変えないでください。修復ノードはこのリポジトリを clone します。Google Drive も外部トンネルも使いません。最後のセルが URL を出します。ノートを開いている同じ Google アカウントのブラウザで開いてください。

## Try it

1. Open the notebook in Colab (Runtime → Change runtime type → **A100**).
2. Run every cell from the top. The first model download is large.
3. The last cell prints a URL. Open it in the **same Google account** that has the notebook open.
4. In ComfyUI, load `workflows/trellis2_pixal3d_workflow.json`.

[Open in Colab](https://colab.research.google.com/github/koba4061/trellis2-pixal3d-comfyui-mesh-repair/blob/main/comfyui_trellis2_pixal3d_image-to-3d_watertight-mesh-repair_colab.ipynb)

The repair node is cloned from this repository. Google Drive is not used. GLB files land in `/content/ComfyUI/output`.

## Why V2

A fresh mesh looks clean and is still a triangle soup: vertices are not welded.

Measured on one Pixal3D GLB (699,458 faces):

| State | boundary | nonmanifold |
|------|---------:|------------:|
| Raw mesh | 269,585 | 12 |
| After weld (1e-5) | 552 | 748 |

Most of the apparent holes are duplicated vertices from UV seams (about 1e-6 apart). Voxel remeshing is not required and would damage the surface.

## Mesh Repair Lite

```
Raw mesh
 → GetMeshInfo
 → RemeshMesh (brief149 settings: udf / qef=false / drops off / smooth=20)
 → DecimateMesh (700,000 faces / midpoint)
 → Mesh Repair Lite
     1. Weld vertices (eps=1e-5)
     2. Decimate (GPU QEM, cluster pre-pass on huge meshes, CPU fallback)
     3. Split connected components, pymeshfix.repair() each one, join again
        (pymeshlab if a component fails)
     4. Optional inner-shell removal (kept if it would delete more than 30%)
     5. Unify normals and write a report
```

Splitting components is required. Passing every part to pymeshfix at once can drop everything except the largest component.

Self-intersection removal was removed. On a hollow object it treats the inner and outer walls as intersections and deletes faces.

Defects that cannot be repaired are still exported, and the report counts them.

### Node parameters

| Parameter | Default | Meaning |
|---|---|---|
| `weld_eps` | 0.00001 | Weld only duplicated vertices. Do not raise this. |
| `input_max_faces` | 5,000,000 | High-poly cap for normal baking. 0 disables the cut. |
| `max_faces` | 700,000 | Final face cap |
| `fill_holes` | true | Fill holes |
| `remove_inner_shell` | false | Delete a shell fully inside another. Aborts if the cut exceeds 30% |

`parts=` in the report is the connected-component count. If it drops, a part was removed at that stage.

## Cells

| Step | What it does |
|---|---|
| 1 | Check that the GPU is an A100 |
| 2 | Note that outputs stay on the Colab disk |
| 3 | Install ComfyUI v0.38.2 without replacing Colab's CUDA torch |
| 3b | Optional SD1.5 smoke-test checkpoint |
| 3.5 | Download TRELLIS.2, Pixal3D, MoGe, and BiRefNet onto the Colab VM |
| 3.7 | Clone this repo and install Mesh Repair Lite. Run this before launch |
| 4 | Start ComfyUI on 127.0.0.1:8188 and print the proxy URL |

## Dependencies

The install cell adds `trimesh`, `scipy`, `pymeshfix`, `pymeshlab`, and `fast-simplification`. GPU decimation uses ComfyUI's own QEM.

## Notes

- A GLB splits vertices on UV seams, so the file can show boundary edges even when the welded geometry is watertight.
- `watertight=True` does not mean the double shell is gone. A closed inner shell is watertight too.
- The proxy URL works only in the browser of the Google account that has this notebook open. It dies when the runtime stops. It is not a public tunnel.

Failure notes: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
