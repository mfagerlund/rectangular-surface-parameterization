"""Test LSCM baseline."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from io_obj import load_obj, save_obj
from lscm import lscm_parameterize, normalize_uvs
from uv_recovery import compute_uv_quality
from visualize import plot_mesh_2d

# Test on sphere (disk topology)
mesh = load_obj("C:/Dev/Colonel/Data/Meshes/sphere320.obj")
print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")

corner_uvs = lscm_parameterize(mesh)
corner_uvs = normalize_uvs(corner_uvs)

quality = compute_uv_quality(mesh, corner_uvs)
print(f"Flipped: {quality['flipped_count']} / {mesh.n_faces}")
print(f"Angle error: {np.degrees(quality['angle_error_mean']):.2f} deg")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_mesh_2d(mesh, corner_uvs, ax=ax, title="LSCM UV Layout")
plt.savefig("lscm_torus.png", dpi=150)
print("Saved: lscm_torus.png")
