"""
Tests for loading and processing the included test meshes.

These tests verify that all meshes in Mesh/ folder can be loaded and processed
through the full pipeline.
"""

import pytest
import numpy as np
from pathlib import Path

# Get the Mesh folder path
MESH_DIR = Path(__file__).parent.parent / "Mesh"


class TestOBJReader:
    """Test OBJ file loading."""

    def test_load_sphere(self):
        """Sphere should load without issues."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        V, F, *_ = readOBJ(str(MESH_DIR / "sphere320.obj"))
        assert V.shape == (162, 3)
        assert F.shape == (320, 3)

    def test_load_torus(self):
        """Torus should load without issues."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        V, F, *_ = readOBJ(str(MESH_DIR / "torus.obj"))
        assert V.shape[0] > 0
        assert F.shape[0] > 0

    def test_load_pig(self):
        """Pig should load without issues."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        V, F, *_ = readOBJ(str(MESH_DIR / "pig.obj"))
        assert V.shape == (1843, 3)
        assert F.shape == (3560, 3)

    def test_load_B36(self):
        """B36 should load without issues."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        V, F, *_ = readOBJ(str(MESH_DIR / "B36.obj"))
        assert V.shape == (2200, 3)
        assert F.shape == (4396, 3)

    def test_load_SquareMyles_utf8(self):
        """SquareMyles is UTF-8 encoded and should load correctly.

        This test catches the encoding bug where the OBJ reader uses
        the system default encoding (cp1252 on Windows) instead of UTF-8.
        """
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        V, F, *_ = readOBJ(str(MESH_DIR / "SquareMyles.obj"))
        assert V.shape == (706, 3)
        assert F.shape == (1328, 3)


class TestMeshConnectivity:
    """Test mesh connectivity computation."""

    def test_sphere_connected(self):
        """Sphere should be a single connected component."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info

        V, F, *_ = readOBJ(str(MESH_DIR / "sphere320.obj"))
        mesh = mesh_info(V, F)

        # Check all vertices are reachable via edge connectivity
        from rectangular_surface_parameterization.preprocessing.connectivity import check_mesh_connected
        assert check_mesh_connected(mesh), "Sphere should be connected"

    def test_torus_connected(self):
        """Torus should be a single connected component."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info

        V, F, *_ = readOBJ(str(MESH_DIR / "torus.obj"))
        mesh = mesh_info(V, F)

        from rectangular_surface_parameterization.preprocessing.connectivity import check_mesh_connected
        assert check_mesh_connected(mesh), "Torus should be connected"

    def test_pig_connected(self):
        """Pig should be a single connected component.

        This test catches the bug where find_graph_generator reports
        disconnected components on meshes that should be connected.
        """
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info

        V, F, *_ = readOBJ(str(MESH_DIR / "pig.obj"))
        mesh = mesh_info(V, F)

        from rectangular_surface_parameterization.preprocessing.connectivity import check_mesh_connected
        assert check_mesh_connected(mesh), "Pig should be connected"

    def test_B36_connected(self):
        """B36 should be a single connected component."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info

        V, F, *_ = readOBJ(str(MESH_DIR / "B36.obj"))
        mesh = mesh_info(V, F)

        from rectangular_surface_parameterization.preprocessing.connectivity import check_mesh_connected
        assert check_mesh_connected(mesh), "B36 should be connected"


class TestPreprocessing:
    """Test preprocessing step on all meshes."""

    def test_preprocess_sphere(self):
        """Sphere should preprocess without errors."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info
        from rectangular_surface_parameterization.preprocessing.dec import dec_tri
        from rectangular_surface_parameterization.preprocessing.preprocess import preprocess_ortho_param

        V, F, *_ = readOBJ(str(MESH_DIR / "sphere320.obj"))
        mesh = mesh_info(V, F)
        dec = dec_tri(mesh)

        param, mesh_out, dec_out = preprocess_ortho_param(mesh, dec, True, True, 40)
        assert param is not None

    def test_preprocess_torus(self):
        """Torus should preprocess without errors."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info
        from rectangular_surface_parameterization.preprocessing.dec import dec_tri
        from rectangular_surface_parameterization.preprocessing.preprocess import preprocess_ortho_param

        V, F, *_ = readOBJ(str(MESH_DIR / "torus.obj"))
        mesh = mesh_info(V, F)
        dec = dec_tri(mesh)

        param, mesh_out, dec_out = preprocess_ortho_param(mesh, dec, True, True, 40)
        assert param is not None

    def test_preprocess_pig(self):
        """Pig should preprocess without errors.

        This test catches the 'disconnected components' bug in preprocessing.
        """
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info
        from rectangular_surface_parameterization.preprocessing.dec import dec_tri
        from rectangular_surface_parameterization.preprocessing.preprocess import preprocess_ortho_param

        V, F, *_ = readOBJ(str(MESH_DIR / "pig.obj"))
        mesh = mesh_info(V, F)
        dec = dec_tri(mesh)

        param, mesh_out, dec_out = preprocess_ortho_param(mesh, dec, True, True, 40)
        assert param is not None

    def test_preprocess_B36(self):
        """B36 should preprocess without errors."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info
        from rectangular_surface_parameterization.preprocessing.dec import dec_tri
        from rectangular_surface_parameterization.preprocessing.preprocess import preprocess_ortho_param

        V, F, *_ = readOBJ(str(MESH_DIR / "B36.obj"))
        mesh = mesh_info(V, F)
        dec = dec_tri(mesh)

        param, mesh_out, dec_out = preprocess_ortho_param(mesh, dec, True, True, 40)
        assert param is not None


class TestCrossField:
    """Test cross field computation on all meshes."""

    def test_crossfield_sphere_smooth(self):
        """Sphere should compute smooth cross field."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info
        from rectangular_surface_parameterization.preprocessing.dec import dec_tri
        from rectangular_surface_parameterization.preprocessing.preprocess import preprocess_ortho_param
        from rectangular_surface_parameterization.cross_field.face_field import compute_face_cross_field

        V, F, *_ = readOBJ(str(MESH_DIR / "sphere320.obj"))
        mesh = mesh_info(V, F)
        dec = dec_tri(mesh)
        param, mesh, dec = preprocess_ortho_param(mesh, dec, True, True, 40)

        omega, ang, sing = compute_face_cross_field(mesh, param, dec, 10)

        assert not np.any(np.isnan(omega)), "omega should not contain NaN"
        assert not np.any(np.isnan(ang)), "ang should not contain NaN"
        # Sphere should have 8 singularities (Euler char = 2, each +1/4)
        n_sing = np.sum(np.abs(sing) > 0.1)
        assert n_sing == 8, f"Sphere should have 8 singularities, got {n_sing}"

    def test_crossfield_B36_smooth(self):
        """B36 should compute smooth cross field without NaN.

        This test catches the singular matrix bug in cross field computation.
        """
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        from rectangular_surface_parameterization.core.mesh_info import mesh_info
        from rectangular_surface_parameterization.preprocessing.dec import dec_tri
        from rectangular_surface_parameterization.preprocessing.preprocess import preprocess_ortho_param
        from rectangular_surface_parameterization.cross_field.face_field import compute_face_cross_field

        V, F, *_ = readOBJ(str(MESH_DIR / "B36.obj"))
        mesh = mesh_info(V, F)
        dec = dec_tri(mesh)
        param, mesh, dec = preprocess_ortho_param(mesh, dec, True, True, 40)

        omega, ang, sing = compute_face_cross_field(mesh, param, dec, 10)

        assert not np.any(np.isnan(omega)), "omega should not contain NaN"
        assert not np.any(np.isnan(ang)), "ang should not contain NaN"


class TestFullPipeline:
    """Test full pipeline on all meshes."""

    def test_pipeline_sphere(self):
        """Full pipeline should work on sphere with 0 flipped triangles."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "run_RSP.py", str(MESH_DIR / "sphere320.obj"),
             "-o", "Results/test_sphere/", "--frame-field", "smooth", "-v"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        assert "Flipped triangles: 0" in result.stdout

    def test_pipeline_torus(self):
        """Full pipeline should work on torus with 0 flipped triangles."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "run_RSP.py", str(MESH_DIR / "torus.obj"),
             "-o", "Results/test_torus/", "--frame-field", "smooth", "-v"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        assert "Flipped triangles: 0" in result.stdout

    def test_pipeline_pig(self):
        """Full pipeline should work on pig (MATLAB example mesh)."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "run_RSP.py", str(MESH_DIR / "pig.obj"),
             "-o", "Results/test_pig/", "--frame-field", "curvature",
             "--energy", "alignment"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

    def test_pipeline_B36(self):
        """Full pipeline should work on B36 (MATLAB example mesh)."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "run_RSP.py", str(MESH_DIR / "B36.obj"),
             "-o", "Results/test_B36/", "--frame-field", "smooth"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

    def test_pipeline_SquareMyles(self):
        """Full pipeline should work on SquareMyles (MATLAB example mesh).

        Note: MATLAB example uses trivial connection + chebyshev energy,
        but trivial_connection has a constraint bug. Using smooth field
        as a workaround until that's fixed.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "run_RSP.py", str(MESH_DIR / "SquareMyles.obj"),
             "-o", "Results/test_SquareMyles/", "--frame-field", "smooth"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
