# TODO

## Add Rendered Images to Documentation

Once the pipeline is fully working, generate images for documentation.

### Pipeline Overview (USAGE.md)
- [ ] `images/pipeline_overview.png` - Diagram: Input Mesh → Cross Field → Cut Graph → UV Layout → Quad Mesh

### Key Concepts (USAGE.md)
- [ ] `images/singularities.png` - Mesh with singularity vertices highlighted (e.g., torus or sphere)
- [ ] `images/uv_seams.png` - UV layout showing seam edges

### Cross Field Comparison (USAGE.md)
Run each on bunny or cow - iconic meshes that show field differences clearly:
```bash
# Using bunny (need to add bunny-small.obj to Mesh/)
python run_RSP.py Mesh/bunny.obj -o images/ --frame-field smooth --save-viz
python run_RSP.py Mesh/bunny.obj -o images/ --frame-field curvature --save-viz
python run_RSP.py Mesh/bunny.obj -o images/ --frame-field trivial --save-viz

# Generate quad meshes to show final result
python extract_quads.py Mesh/bunny.obj -o images/ --scale 10 --frame-field smooth
python extract_quads.py Mesh/bunny.obj -o images/ --scale 10 --frame-field curvature
python extract_quads.py Mesh/bunny.obj -o images/ --scale 10 --frame-field trivial
```
- [ ] `images/crossfield_smooth.png` - Smooth cross field quad mesh (bunny/cow)
- [ ] `images/crossfield_curvature.png` - Curvature-aligned quad mesh (bunny/cow)
- [ ] `images/crossfield_trivial.png` - Trivial connection quad mesh (bunny/cow)
- [ ] `images/crossfield_comparison.png` - Side-by-side 3-up comparison

**Why bunny/cow?** These meshes have recognizable features (ears, legs) that make
it easy to see how different cross field types affect quad orientation.

### Energy Type Comparison (USAGE.md)
Run each on the same mesh:
```bash
python run_RSP.py Mesh/pig.obj -o images/ --energy distortion --w-conf-ar 0.0 --save-viz
python run_RSP.py Mesh/pig.obj -o images/ --energy distortion --w-conf-ar 0.5 --save-viz
python run_RSP.py Mesh/pig.obj -o images/ --energy distortion --w-conf-ar 1.0 --save-viz
python run_RSP.py Mesh/pig.obj -o images/ --energy chebyshev --save-viz
python run_RSP.py Mesh/pig.obj -o images/ --energy alignment --save-viz
```
- [ ] `images/energy_distortion_comparison.png` - w-conf-ar 0.0 vs 0.5 vs 1.0
- [ ] `images/energy_chebyshev.png` - Chebyshev net result
- [ ] `images/energy_alignment.png` - Alignment energy result
- [ ] `images/energy_comparison.png` - Side-by-side comparison (composite)

### Hero Image (README.md)
- [ ] `images/hero_example.png` - Before/after: triangle mesh → quad mesh

### Quad Extraction Examples
```bash
python extract_quads.py Mesh/sphere320.obj -o images/ --scale 10
python extract_quads.py Mesh/torus.obj -o images/ --scale 10
python extract_quads.py Mesh/pig.obj -o images/ --scale 10
```
- [ ] `images/quads_sphere.png` - Sphere quad mesh
- [ ] `images/quads_torus.png` - Torus quad mesh
- [ ] `images/quads_pig.png` - Pig quad mesh

### Per-Mesh Gallery
For each included mesh, show: input mesh, UV layout, quad result
- [ ] sphere320: input, UV, quads
- [ ] torus: input, UV, quads
- [ ] pig: input, UV, quads
- [ ] B36: input, UV, quads
- [ ] SquareMyles: input, UV, quads

---

## Fixed Issues

These bugs were blocking test meshes and have been fixed:
- [x] `find_graph_generator` disconnected components (pig, SquareMyles) - scipy treats weight=0 as "no edge" while MATLAB treats it as "zero cost". Fixed by using small positive weight (1e-10) for boundary edges.
- [x] Singular matrix in cross field computation (B36) - Added `regularized_solve()` function that provides MATLAB-like robustness for singular systems.
- [x] Open meshes failing at reduce_corner_var_2d (pig, SquareMyles) - Now automatically uses `reduce_corner_var_2d_cut` for meshes with boundary edges.

## Remaining Issues

- [ ] `trivial_connection` assertion failure - SquareMyles with trivial field fails on "Failed to prescribe constraints between feature curves". Workaround: use smooth field.
- [ ] Optimization convergence - pig has 70 flipped triangles, SquareMyles has 2. Likely needs regularization in `optimize_RSP.py`.

## Notes

- Use consistent camera angles for comparison shots
- Consider using a render script to batch-generate all images
- Images should be ~800px wide for good README display
- Use PNG format for crisp edges on mesh renders
