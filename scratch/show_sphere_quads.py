"""Visualize quad mesh extracted from sphere320 golden data."""

import sys
sys.path.insert(0, r"C:\Dev\rectangular-surface-parameterization")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rectangular_surface_parameterization.utils.quad_extractor import extract_quads

# Load golden parameterization
data = np.load(r"C:\Dev\rectangular-surface-parameterization\tests\golden_data\sphere320_param.npz")
V, T, uv = data["vertices"], data["triangles"], data["uv_per_tri"]

# Extract quads at scale=10
uv_scaled = uv * 10.0
qv, qf, tf = extract_quads(V, T, uv_scaled, verbose=True, fill_holes=False)
print(f"\nResult: {len(qv)} vertices, {len(qf)} quads")

# Plot
fig = plt.figure(figsize=(14, 6))

# --- Left: wireframe ---
ax1 = fig.add_subplot(121, projection="3d")
for q in qf:
    corners = qv[q]
    loop = np.vstack([corners, corners[0:1]])
    ax1.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="steelblue", lw=0.6)
ax1.set_title(f"Quad wireframe ({len(qf)} quads)")
ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(-1.1, 1.1); ax1.set_zlim(-1.1, 1.1)
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# --- Right: shaded faces ---
ax2 = fig.add_subplot(122, projection="3d")
polys = []
colors = []
for q in qf:
    corners = qv[q]
    polys.append(corners)
    # Color by face normal dot with light direction
    e1 = corners[1] - corners[0]
    e2 = corners[3] - corners[0]
    n = np.cross(e1, e2)
    n = n / (np.linalg.norm(n) + 1e-12)
    light = np.array([0.3, 0.5, 0.8])
    light /= np.linalg.norm(light)
    shade = 0.3 + 0.7 * max(0, np.dot(n, light))
    colors.append([0.3 * shade, 0.55 * shade, 0.85 * shade, 0.9])

pc = Poly3DCollection(polys, facecolors=colors, edgecolors="k", linewidths=0.3)
ax2.add_collection3d(pc)
ax2.set_xlim(-1.1, 1.1); ax2.set_ylim(-1.1, 1.1); ax2.set_zlim(-1.1, 1.1)
ax2.set_box_aspect([1, 1, 1])
ax2.set_title(f"Shaded quads ({len(qv)} verts)")
ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

plt.tight_layout()
out = r"C:\Dev\rectangular-surface-parameterization\scratch\sphere_quads_viz.png"
plt.savefig(out, dpi=180)
print(f"Saved to {out}")
plt.show()
