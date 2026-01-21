"""Debug LSCM implementation."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, lsqr
from io_obj import load_obj

mesh = load_obj("C:/Dev/Colonel/Data/Meshes/sphere320.obj")
print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")

n_faces = mesh.n_faces
n_vertices = mesh.n_vertices

# Build LSCM matrices
A_real_rows = []
A_real_cols = []
A_real_data = []
A_imag_rows = []
A_imag_cols = []
A_imag_data = []

for f in range(n_faces):
    v0, v1, v2 = mesh.faces[f]
    p0 = mesh.positions[v0]
    p1 = mesh.positions[v1]
    p2 = mesh.positions[v2]

    e01 = p1 - p0
    e02 = p2 - p0

    len01 = np.linalg.norm(e01)
    if len01 < 1e-10:
        continue
    x_dir = e01 / len01

    normal = np.cross(e01, e02)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-10:
        continue
    normal = normal / normal_len
    y_dir = np.cross(normal, x_dir)

    x0, y0 = 0.0, 0.0
    x1, y1 = len01, 0.0
    x2, y2 = np.dot(e02, x_dir), np.dot(e02, y_dir)

    c0_re = x1 - x2
    c0_im = y2 - y1  # = y2 since y1 = 0
    c1_re = x2
    c1_im = -y2
    c2_re = -x1
    c2_im = y1  # = 0

    # Real equation
    A_real_rows.extend([f, f, f, f, f, f])
    A_real_cols.extend([v0, n_vertices + v0, v1, n_vertices + v1, v2, n_vertices + v2])
    A_real_data.extend([c0_re, -c0_im, c1_re, -c1_im, c2_re, -c2_im])

    # Imaginary equation
    A_imag_rows.extend([f, f, f, f, f, f])
    A_imag_cols.extend([v0, n_vertices + v0, v1, n_vertices + v1, v2, n_vertices + v2])
    A_imag_data.extend([c0_im, c0_re, c1_im, c1_re, c2_im, c2_re])

A_real = sparse.csr_matrix((A_real_data, (A_real_rows, A_real_cols)),
                            shape=(n_faces, 2 * n_vertices))
A_imag = sparse.csr_matrix((A_imag_data, (A_imag_rows, A_imag_cols)),
                            shape=(n_faces, 2 * n_vertices))
A = sparse.vstack([A_real, A_imag])

print(f"\nA shape: {A.shape}")
print(f"A nnz: {A.nnz}")
print(f"A_real nnz: {A_real.nnz}")
print(f"A_imag nnz: {A_imag.nnz}")

# Check A^T A
ATA = A.T @ A
print(f"\nA^T A shape: {ATA.shape}")
print(f"A^T A diagonal range: [{ATA.diagonal().min():.6f}, {ATA.diagonal().max():.6f}]")

# Check coupling between u and v blocks
# u block: columns 0 to n_vertices-1
# v block: columns n_vertices to 2*n_vertices-1
ATA_uu = ATA[:n_vertices, :n_vertices]
ATA_vv = ATA[n_vertices:, n_vertices:]
ATA_uv = ATA[:n_vertices, n_vertices:]

print(f"\nATA_uu (u-u coupling) nnz: {ATA_uu.nnz}")
print(f"ATA_vv (v-v coupling) nnz: {ATA_vv.nnz}")
print(f"ATA_uv (u-v coupling) nnz: {ATA_uv.nnz}")
print(f"ATA_uv max: {np.abs(ATA_uv.data).max() if ATA_uv.nnz > 0 else 0:.6f}")

# Find two far apart vertices
dists = np.linalg.norm(mesh.positions[:, np.newaxis, :] - mesh.positions[np.newaxis, :, :], axis=2)
i, j = np.unravel_index(np.argmax(dists), dists.shape)
print(f"\nPinned vertices: {i} and {j} (distance: {dists[i,j]:.4f})")

# Build constraint matrix
C_rows = [0, 1, 2, 3]
C_cols = [i, n_vertices + i, j, n_vertices + j]
C_data = [1.0, 1.0, 1.0, 1.0]
C = sparse.csr_matrix((C_data, (C_rows, C_cols)), shape=(4, 2 * n_vertices))
c_rhs = np.array([0.0, 0.0, 1.0, 0.0])

# Build KKT system
K = sparse.bmat([
    [ATA, C.T],
    [C, sparse.csr_matrix((4, 4))]
])
rhs = np.concatenate([np.zeros(2 * n_vertices), c_rhs])

print(f"\nKKT system size: {K.shape}")
print(f"KKT diagonal range: [{K.diagonal().min():.6f}, {K.diagonal().max():.6f}]")
print(f"KKT diagonal zeros: {np.sum(np.abs(K.diagonal()) < 1e-10)}")

# Try different regularization levels
for reg in [1e-10, 1e-8, 1e-6, 1e-4]:
    K_reg = K + reg * sparse.eye(K.shape[0])
    sol = spsolve(K_reg.tocsc(), rhs)
    x = sol[:2 * n_vertices]
    u = x[:n_vertices]
    v = x[n_vertices:]

    residual = np.linalg.norm(A @ x)
    constraint_err = np.linalg.norm(C @ x - c_rhs)

    print(f"\nReg={reg:.0e}: u range=[{u.min():.4f}, {u.max():.4f}], v range=[{v.min():.6f}, {v.max():.6f}]")
    print(f"  |Ax|={residual:.4f}, |Cx-c|={constraint_err:.6f}")
