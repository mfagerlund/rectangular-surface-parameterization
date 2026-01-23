# === ISSUES ===
# - None: straightforward file writing
# === END ISSUES ===

import numpy as np
from typing import Optional
import warnings


# function writeObj(filename, V,F,UV,TF,N,NF,E2V)
#   % WRITEOBJ writes an OBJ file with vertex/face information
#   %
#   % writeOBJ(filename,V,F,UV,N)
#   %
#   % Input:
#   %  filename  path to .obj file
#   %  V  #V by 3 list of vertices
#   %  F  #F by 3 list of triangle indices
#   %  UV  #UV by 2 list of texture coordinates
#   %  TF  #TF by 3 list of corner texture indices into UV
#   %  N  #N by 3 list of normals
#   %  NF  #NF by 3 list of corner normal indices into N
#   %  E2V  #NE by 2 list of vertices indices of feature edge

def writeObj(
    filename: str,
    V: np.ndarray,
    F: np.ndarray,
    UV: Optional[np.ndarray] = None,
    TF: Optional[np.ndarray] = None,
    N: Optional[np.ndarray] = None,
    NF: Optional[np.ndarray] = None,
    E2V: Optional[np.ndarray] = None
) -> None:
    """
    Write an OBJ file with vertex/face information.

    Args:
        filename: path to .obj file
        V: vertices (#V, 2 or 3)
        F: face indices (#F, 3), 0-indexed
        UV: texture coordinates (#UV, 2), optional
        TF: face texture indices (#F, 3), 0-indexed, optional
        N: normals (#N, 3), optional
        NF: face normal indices (#F, 3), 0-indexed, optional
        E2V: feature edge vertex indices (#NE, 2), 0-indexed, optional

    Note: Python uses 0-indexed faces; OBJ format uses 1-indexed, so we add 1 when writing.
    """
    # if size(V,2) == 2
    #   warning('Appending 0s as z-coordinate');
    #   V(:,end+1:3) = 0;
    # else
    #   assert(size(V,2) == 3);
    # end

    if V.shape[1] == 2:
        warnings.warn('Appending 0s as z-coordinate')
        V = np.hstack([V, np.zeros((V.shape[0], 1))])
    else:
        assert V.shape[1] == 3

    # hasN =  exist('N','var') && ~isempty(N);
    # hasUV = exist('UV','var') && ~isempty(UV);
    # hasE2V = exist('E2V','var') && ~isempty(E2V);

    hasN = N is not None and len(N) > 0
    hasUV = UV is not None and len(UV) > 0
    hasE2V = E2V is not None and len(E2V) > 0

    # if hasUV && (~exist('TF','var') || isempty(TF))
    #     TF = F;
    # end
    # if hasN && (~exist('NF','var') || isempty(NF))
    #     NF = F;
    # end

    if hasUV and (TF is None or len(TF) == 0):
        TF = F.copy()
    if hasN and (NF is None or len(NF) == 0):
        NF = F.copy()

    # f = fopen( filename, 'w' );

    with open(filename, 'w') as f:
        # if ~isempty(V)
        #     fprintf( f, 'v %g %g %g\n', V');
        # end

        # Write vertices
        if len(V) > 0:
            for v in V:
                f.write(f'v {v[0]:g} {v[1]:g} {v[2]:g}\n')

        # if hasUV
        #     fprintf( f, 'vt %0.17g %0.17g\n', UV(:,1:2)');
        # end

        # Write texture coordinates
        if hasUV:
            for uv in UV:
                f.write(f'vt {uv[0]:.17g} {uv[1]:.17g}\n')

        # if hasN
        #     fprintf( f, 'vn %0.17g %0.17g %0.17g\n', N');
        # end

        # Write normals
        if hasN:
            for n in N:
                f.write(f'vn {n[0]:.17g} {n[1]:.17g} {n[2]:.17g}\n')

        # for k=1:size(F,1)
        #     if ( (~hasN) && (~hasUV) ) || (any(TF(k,:)<=0,2) && any(NF(k,:)<=0,2))
        #         fmt = repmat(' %d',1,size(F,2));
        #         fprintf( f,['f' fmt '\n'], F(k,:));
        #     elseif ( hasUV && (~hasN || any(NF(k,:)<=0,2)))
        #         fmt = repmat(' %d/%d',1,size(F,2));
        #         fprintf( f, ['f' fmt '\n'], [F(k,:);TF(k,:)]);
        #     elseif ( (hasN) && (~hasUV || any(TF(k,:)<=0,2)))
        #         fmt = repmat(' %d//%d',1,size(F,2));
        #         fprintf( f, ['f' fmt '\n'],[F(k,:);TF(k,:)]');
        #     elseif ( (hasN) && (hasUV) )
        #         fmt = repmat(' %d/%d/%d',1,size(F,2));
        #         fprintf( f, ['f' fmt '\n'],[F(k,:);TF(k,:);NF(k,:)]);
        #     end
        # end

        # Write faces (convert from 0-indexed to 1-indexed for OBJ format)
        for k in range(F.shape[0]):
            # Get face indices (skip padding -1 values for quads that are triangles)
            face_indices = F[k, :]
            valid_mask = face_indices >= 0
            valid_face = face_indices[valid_mask]

            if len(valid_face) == 0:
                continue

            # Check conditions for format selection
            # In Python: index >= 0 means valid (MATLAB used > 0 for 1-indexed)
            # TF(k,:)<=0 in MATLAB corresponds to TF[k,:] < 0 in Python (boundary sentinel -1)
            tf_invalid = TF is None or np.any(TF[k, valid_mask] < 0)
            nf_invalid = NF is None or np.any(NF[k, valid_mask] < 0)

            if (not hasN and not hasUV) or (tf_invalid and nf_invalid):
                # f v1 v2 v3
                parts = ' '.join(str(vi + 1) for vi in valid_face)
                f.write(f'f {parts}\n')
            elif hasUV and (not hasN or nf_invalid):
                # f v1/vt1 v2/vt2 v3/vt3
                tf_indices = TF[k, valid_mask]
                parts = ' '.join(f'{valid_face[i] + 1}/{tf_indices[i] + 1}' for i in range(len(valid_face)))
                f.write(f'f {parts}\n')
            elif hasN and (not hasUV or tf_invalid):
                # f v1//vn1 v2//vn2 v3//vn3
                nf_indices = NF[k, valid_mask]
                parts = ' '.join(f'{valid_face[i] + 1}//{nf_indices[i] + 1}' for i in range(len(valid_face)))
                f.write(f'f {parts}\n')
            elif hasN and hasUV:
                # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                tf_indices = TF[k, valid_mask]
                nf_indices = NF[k, valid_mask]
                # assert(all(NF(k,:)>0));
                # assert(all(TF(k,:)>0));
                assert np.all(nf_indices >= 0), f"Invalid normal indices at face {k}"
                assert np.all(tf_indices >= 0), f"Invalid texture indices at face {k}"
                parts = ' '.join(f'{valid_face[i] + 1}/{tf_indices[i] + 1}/{nf_indices[i] + 1}' for i in range(len(valid_face)))
                f.write(f'f {parts}\n')

        # % print feature edges
        # if hasE2V
        #     fprintf( f, 'l %d %d\n', E2V');
        # end

        # Write feature edges (convert to 1-indexed)
        if hasE2V:
            for e in E2V:
                f.write(f'l {e[0] + 1} {e[1] + 1}\n')

    # fclose(f);
