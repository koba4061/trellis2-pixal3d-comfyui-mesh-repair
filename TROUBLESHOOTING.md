# Mesh repair pipeline — what failed, and why

How a TRELLIS.2 / Pixal3D mesh was made topologically cleaner without throwing away the surface.

## Goal

Raw meshes have unwelded vertices, non-manifold edges, double shells, self-intersections, and holes.

Target: watertight, no non-manifold edges, no double shell, no self-intersection, same look.

Final chain:

```
Raw mesh (tens of millions of faces, inconsistent winding)
 → GetMeshInfo
 → RemeshMesh (brief149: udf / qef=false / drops off / smooth_iters=20)
 → DecimateMesh (700,000 faces / midpoint)
 → Mesh Repair Lite (weld → GPU QEM → per-component pymeshfix → optional inner shell → normals)
```

RemeshMesh and DecimateMesh stay on the original settings. An earlier V2 set `qef=true`, `smooth_iters=3`, and DecimateMesh to 5M/qem. That is a likely cause of lost thin parts such as wings.

Self-intersection removal was deleted. It destroys hollow walls.

## What failed

### 1. Voxel / SDF rebuild

The output became a lattice of holes. Voxel closing rebuilds the shape and drops thin parts and interior structure. The smallest repair that does not rebuild the surface is the one that keeps the look.

### 2. pymeshfix on the raw mesh

A 92M-face mesh, still millions of non-manifold edges after welding, ran for many hours. One run collapsed to 6 faces. Defect-by-defect repair diverges on a dense broken mesh. QEM on that input fragments it, and `repair()` then deletes almost every component.

### 3. CPU decimate only

fast-simplification stopped at 4.26M faces instead of 700k. Boundary-heavy meshes converge poorly on CPU.

### 4. Bypassed nodes (`mode: 4`)

RemeshMesh and DecimateMesh stayed muted (purple) while the wires looked connected, so the raw mesh skipped ahead. A purple node is disabled. If GetMeshInfo still reports tens of millions of faces, the upstream nodes are not running.

### 5. `sign_mode: sdf` on RemeshMesh

SDF needs consistent winding. The raw mesh does not have it, so inside and outside flip and the surface extraction breaks. `udf` is the mode marked robust to messy, non-manifold input. UDF also builds a thin closed shell from both sides of a thin sheet, which can leave an inner shell.

### 6. pymeshfix deletes whole parts

A butterfly wing on a jewelry box disappeared (697,754 → 384,468 faces, about 45%). `MeshFix.repair()` defaults to `remove_smallest_components=True` and keeps only the largest component. The wing was a separate component joined at the hinge.

Passing `remove_smallest_components=False` was not enough in this environment. The installed pymeshfix rejected the argument (TypeError) and fell back to the default, so parts still went from 27 to 1.

The fix is structural: split into connected components, `repair()` each one, then join. A repair of a single component cannot delete the others. A failed component is repaired with pymeshlab, or kept as-is.

### 7. Intersection removal eats hollow walls

Turning component deletion off made the jewelry box shatter. A hollow box from UDF has an inner wall and an outer wall close together. `remove_intersections` treated those walls as self-intersections and deleted faces. The option was removed. Internal intersections do not show in the shaded view, and the damage is worse than the defect.

Do not enable several "delete" options at once. Turn one on, read `parts=`, then decide.

### 8. QEF and a short smooth

`qef=true`, `smooth_iters=3`, and DecimateMesh at 5M/qem differ from the brief149 settings (`qef=false`, smooth 20, 700k/midpoint). Thin parts are sensitive to that difference. The node restores the original settings. Even then, pymeshfix was the part that deleted components, not the remesh.

## What worked

### Weld (`eps=1e-5`)

UV export duplicates vertices by about 1e-6. Welding them removed most boundary edges in one measurement (269,585 → 552) without changing the look. Do this first. Before the weld, `nonmanifold=0` is not a pass: split vertices hide the real T-junctions. After the weld those edges become visible, and the repair is what drives them to 0.

### Original RemeshMesh + DecimateMesh

`sign_mode=udf`, `qef=false`, drops off, `smooth_iters=20`, then DecimateMesh at 700,000 / midpoint. Mesh Repair Lite sits after DecimateMesh so pymeshfix sees about 700k faces.

### Per-component repair

This is what keeps wings, hinges, and trim. One `repair()` call on a multi-component mesh can keep only the largest piece.

### Two-step GPU QEM

At or under 5M faces, QEM runs directly. Larger inputs get a cluster pre-pass, then QEM. If the GPU path fails, CPU fast-simplification finishes. The final cut runs from the already reduced mesh, not from the original twice.

### Collapse guard

If pymeshfix output is under 5% of the input and under 10 faces, that component is a failure and pymeshlab takes over. `watertight=True` on a 6-face remnant is not success. The test is a ratio, so a small part that stays the same size is not flagged.

## Operating notes

- Read GetMeshInfo. A few million faces means remesh and decimate ran. Tens of millions means a bypass.
- Do not treat `watertight=True` alone as success. Also check that the face count did not collapse, and that `parts=` did not drop a piece you need.
- A closed inner shell is watertight. Inner-shell removal only deletes a component that sits fully inside another, and it keeps everything if the cut would exceed 30%.
- `smooth_iters` stays at 20 unless you retest `qef` and the drop flags together.
- pymeshlab can abort inside C++ with no Python traceback on a huge broken input.
- A thin wing or cloth becoming a thin shell under UDF is expected. Deleting that shell opens a hole.

## Defect classes

| Problem | Meaning | This pipeline |
|---|---|---|
| Boundary edge | An edge with one face. A hole. Not watertight. | Repaired |
| Non-manifold | An edge shared by 3 or more faces | Repaired |
| Inconsistent winding | Flipped faces. Breaks SDF. | Normal unify |
| Degenerate face | Zero area | Weld + repair |
| Duplicate vertices | Same position, split apart | Weld |
| Self-intersection | Faces pass through each other | Not removed. It deletes real walls. |
| Double shell | A shell inside the outer surface | Optional. Not fully removed. |
| Loose part | A separate component | Kept, so wings are not deleted |

A GLB still shows boundary edges after a watertight repair because UV seams duplicate vertices. Weld on import if you need to check the geometry in Blender.
