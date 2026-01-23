# === ISSUES ===
# - system call to QuantizationYoann: external program, not implemented
# === END ISSUES ===

import numpy as np
import os
from typing import Optional
import warnings
import subprocess

from Utils.writeObj import writeObj


# function save_param(ifquantization, path_save, mesh_name, X, T, UV, TUV, sing, E2V_hardedge)

def save_param(
    ifquantization: bool,
    path_save: str,
    mesh_name: str,
    X: np.ndarray,
    T: np.ndarray,
    UV: np.ndarray,
    TUV: np.ndarray,
    sing: np.ndarray,
    E2V_hardedge: Optional[np.ndarray] = None
) -> None:
    """
    Save parameterization results to OBJ files.

    Saves three OBJ files:
    - {mesh_name}_pos.obj: vertices with positive singularity index (sing > 1/8)
    - {mesh_name}_neg.obj: vertices with negative singularity index (sing < -1/8)
    - {mesh_name}_param.obj: full parameterized mesh with UVs

    Optionally runs quantization (requires external QuantizationYoann program).

    Args:
        ifquantization: whether to run quantization step
        path_save: directory to save output files
        mesh_name: base name for output files
        X: vertex positions (#V, 3)
        T: face indices (#F, 3), 0-indexed
        UV: texture coordinates (#UV, 2)
        TUV: face texture indices (#F, 3), 0-indexed
        sing: singularity indices per vertex (#V,)
        E2V_hardedge: hard edge vertex pairs (#E, 2), 0-indexed, optional
    """
    # if ~exist(path_save, 'dir')
    #     mkdir(path_save);
    # end

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # id_sing_p = sing > 1/8;
    # id_sing_m = sing <-1/8;

    id_sing_p = sing > 1/8
    id_sing_m = sing < -1/8

    # writeObj([path_save, mesh_name, '_pos.obj'], X(id_sing_p,:), []);

    # Write positive singularities (just vertices, no faces)
    pos_path = os.path.join(path_save, f'{mesh_name}_pos.obj')
    writeObj(pos_path, X[id_sing_p, :], np.zeros((0, 3), dtype=int))

    # writeObj([path_save, mesh_name, '_neg.obj'], X(id_sing_m,:), []);

    # Write negative singularities (just vertices, no faces)
    neg_path = os.path.join(path_save, f'{mesh_name}_neg.obj')
    writeObj(neg_path, X[id_sing_m, :], np.zeros((0, 3), dtype=int))

    # writeObj([path_save, mesh_name, '_param.obj'], X, T, UV, TUV, [], [], E2V_hardedge);

    # Write full parameterized mesh
    param_path = os.path.join(path_save, f'{mesh_name}_param.obj')
    writeObj(param_path, X, T, UV, TUV, None, None, E2V_hardedge)

    # if ifquantization
    #     if  exist('./QuantizationYoann/build/Quantization', 'file') ~= 0
    #         status = system(['./QuantizationYoann/build/Quantization -s a -sa ', num2str(1), ' -r -o ', path_save, mesh_name, '_quantiz.obj ', path_save, mesh_name, '_param.obj']);
    #         if status ~= 0
    #             warning('Quantization: Yoann failed me :(');
    #         end
    #     else
    #         error('Must compile the Quantization program. Go to folder QuantizationYoann/');
    #     end
    # end

    if ifquantization:
        quantize_exe = './QuantizationYoann/build/Quantization'

        if os.path.exists(quantize_exe):
            quantiz_path = os.path.join(path_save, f'{mesh_name}_quantiz.obj')
            cmd = [
                quantize_exe,
                '-s', 'a',
                '-sa', '1',
                '-r',
                '-o', quantiz_path,
                param_path
            ]

            try:
                result = subprocess.run(cmd, check=False)
                if result.returncode != 0:
                    warnings.warn('Quantization: Yoann failed me :(')
            except Exception as e:
                warnings.warn(f'Quantization failed: {e}')
        else:
            raise RuntimeError('Must compile the Quantization program. Go to folder QuantizationYoann/')
