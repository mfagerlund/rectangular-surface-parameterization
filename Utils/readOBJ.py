# === ISSUES ===
# - None: straightforward file parsing
# === END ISSUES ===

import numpy as np
from typing import Tuple, Optional
import warnings


# function [V,F,UV,TF,N,NF,SI] = readOBJ(filename,varargin)
#   % READOBJ reads an OBJ file with vertex/face information
#   %
#   % [V,F,UV,TF,N,NF] = readOBJ(filename)
#   % [V,F,UV,TF,N,NF] = readOBJ(filename,'ParameterName',ParameterValue,...)
#   %
#   % Input:
#   %  filename  path to .obj file
#   %  Optional:
#   %    'Quads' whether to output face information in X by 4 matrices (faces
#   %      with degree larger than 4 are still triangulated). A trailing zero
#   %      will mean a triangle was read.
#   % Outputs:
#   %  V  #V by 3 list of vertices
#   %  F  #F by 3 list of triangle indices
#   %  UV  #V by 2 list of texture coordinates
#   %  TF  #F by 3 list of triangle texture coordinates
#   %  N  #V by 3 list of normals
#   %  NF  #F by 3 list of triangle corner normal indices into N
#   %
#   % See also: load_mesh, readOBJfast, readOFF

def readOBJ(filename: str, quads: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an OBJ file with vertex/face information.

    Args:
        filename: path to .obj file
        quads: whether to output face information in X by 4 matrices

    Returns:
        V: vertices (#V, 3)
        F: face indices (#F, 3 or 4), 0-indexed
        UV: texture coordinates (#UV, 2 or 3)
        TF: face texture indices (#F, 3 or 4), 0-indexed
        N: normals (#N, 3)
        NF: face normal indices (#F, 3 or 4), 0-indexed
        SI: singularity info (#SI, 2)

    Note: MATLAB returns 1-indexed faces; Python returns 0-indexed faces.
    """
    # % simplex size
    # if quads
    #     ss = 4;
    # else
    #     ss = 3;
    # end

    # simplex size
    ss = 4 if quads else 3

    # % Amortized array allocation
    # Use lists for dynamic allocation in Python

    # Amortized array allocation
    V_list = []
    F_list = []
    UV_list = []
    TF_list = []
    N_list = []
    NF_list = []
    SI_list = []

    # Track counts
    numv = 0
    numuv = 0
    numn = 0

    # triangulated = false;
    # all_ss = true;

    triangulated = False
    all_ss = True

    # fp = fopen( filename, 'r' );
    # type = fscanf( fp, '%s', 1 );
    # while strcmp( type, '' ) == 0

    with open(filename, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            type_token = parts[0]

            # if strcmp( type, 'v' ) == 1
            #     v = sscanf( line, '%lf %lf %lf' );
            #     numv = numv+1;
            #     if length(v) == 2
            #       V(numv,1:2) = [v(1:2)'];
            #     else
            #       V(numv,:) = [v(1:3)'];
            #     end

            if type_token == 'v':
                v = [float(x) for x in parts[1:4]]
                if len(v) == 2:
                    v.append(0.0)
                elif len(v) < 2:
                    continue
                V_list.append(v[:3])
                numv += 1

            # elseif strcmp( type, 'vt')
            #     v = sscanf( line, '%f %f %f' );
            #     numuv = numuv+1;
            #     UV(numuv,:) = [v'];

            elif type_token == 'vt':
                v = [float(x) for x in parts[1:]]
                if len(v) >= 2:
                    UV_list.append(v[:min(3, len(v))])
                    numuv += 1

            # elseif strcmp( type, 'vn')
            #     n = sscanf( line, '%f %f %f' );
            #     numn = numn+1;
            #     N(numn,:) = [n'];

            elif type_token == 'vn':
                n = [float(x) for x in parts[1:4]]
                if len(n) >= 3:
                    N_list.append(n[:3])
                    numn += 1

            # elseif strcmp( type, 'f' ) == 1
            #     [t, count] = sscanf(line,'%d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d');
            #     ... (various formats)

            elif type_token == 'f':
                # Parse face - handle v, v/vt, v/vt/vn, v//vn formats
                face_v = []
                face_vt = []
                face_vn = []

                for vertex in parts[1:]:
                    indices = vertex.split('/')

                    # Vertex index (required)
                    vi = int(indices[0])
                    # Handle negative indices (relative to current count)
                    # t = t + (t<0).*   (numv+1);
                    if vi < 0:
                        vi = numv + vi + 1
                    face_v.append(vi)

                    # Texture coordinate index (optional)
                    vti = -1
                    if len(indices) > 1 and indices[1]:
                        vti = int(indices[1])
                        # tf = tf + (tf<0).*(numuv+1);
                        if vti < 0:
                            vti = numuv + vti + 1
                    face_vt.append(vti)

                    # Normal index (optional)
                    vni = -1
                    if len(indices) > 2 and indices[2]:
                        vni = int(indices[2])
                        # nf = nf + (nf<0).*(numn+1);
                        if vni < 0:
                            vni = numn + vni + 1
                    face_vn.append(vni)

                # Convert to numpy arrays for processing
                t = np.array(face_v)
                tf = np.array(face_vt)
                nf = np.array(face_vn)

                # if numel(t) > ss
                #   if ~triangulated
                #     warning('Trivially triangulating high degree facets');
                #   end
                #   triangulated = true;
                # end

                if len(t) > ss:
                    if not triangulated:
                        warnings.warn('Trivially triangulating high degree facets')
                    triangulated = True

                # Fan triangulation for polygons with > ss vertices
                # while true
                #   if numel(t) > ss
                #     corners = [1 2 3];  (MATLAB 1-indexed)
                #   else
                #     corners = 1:numel(t);
                #   end
                #   ...
                #   if numel(t) <= ss
                #     break;
                #   end
                #   t = t([1 3:end]);

                while True:
                    if len(t) > ss:
                        corners = [0, 1, 2]  # 0-indexed
                    else:
                        # if all_ss && numel(t)<ss
                        #   warning('Small degree facet found');
                        #   all_ss = false;
                        # end
                        if all_ss and len(t) < ss:
                            warnings.warn('Small degree facet found')
                            all_ss = False
                        corners = list(range(len(t)))

                    # Store face (convert to 0-indexed: subtract 1)
                    # F(numf,1:numel(corners)) = [t(corners)'];
                    face = [t[c] - 1 for c in corners]
                    # Pad with -1 for triangles when using quads (MATLAB uses 0, Python uses -1)
                    while len(face) < ss:
                        face.append(-1)
                    F_list.append(face)

                    # Store texture face
                    # TF(numtf,1:numel(corners)) = [tf(corners)'];
                    tface = [tf[c] - 1 if tf[c] > 0 else -1 for c in corners]
                    while len(tface) < ss:
                        tface.append(-1)
                    TF_list.append(tface)

                    # Store normal face
                    # NF(numnf,1:numel(corners)) = [nf(corners)'];
                    nface = [nf[c] - 1 if nf[c] > 0 else -1 for c in corners]
                    while len(nface) < ss:
                        nface.append(-1)
                    NF_list.append(nface)

                    # if numel(t) <= ss
                    #   break;
                    # end
                    if len(t) <= ss:
                        break

                    # t = t([1 3:end]); - fan triangulation
                    t = np.concatenate([[t[0]], t[2:]])
                    tf = np.concatenate([[tf[0]], tf[2:]])
                    nf = np.concatenate([[nf[0]], nf[2:]])

                    if len(t) < 3:
                        break

            # elseif strcmp( type, 'c')
            #     s = sscanf( line, '%d %f' );
            #     numc = numc+1;
            #     SI(numc,:) = [s'];

            elif type_token == 'c':
                s = [float(x) for x in parts[1:3]]
                if len(s) >= 2:
                    SI_list.append(s[:2])

            # elseif strcmp( type, '#' ) == 1
            #     % ignore line

            elif type_token == '#':
                # ignore comment line
                pass

    # V = V(1:numv,:);
    # F = F(1:numf,:);
    # UV = UV(1:numuv,:);
    # TF = TF(1:numtf,:);
    # N = N(1:numn,:);
    # NF = NF(1:numnf,:);
    # SI = SI(1:numc,:);

    # Convert lists to numpy arrays
    V = np.array(V_list) if V_list else np.zeros((0, 3))
    F = np.array(F_list, dtype=int) if F_list else np.zeros((0, ss), dtype=int)

    # Handle UV dimensions
    if UV_list:
        max_uv_dim = max(len(uv) for uv in UV_list)
        UV = np.zeros((len(UV_list), max_uv_dim))
        for i, uv in enumerate(UV_list):
            UV[i, :len(uv)] = uv
    else:
        UV = np.zeros((0, 2))

    TF = np.array(TF_list, dtype=int) if TF_list else np.zeros((0, ss), dtype=int)
    N = np.array(N_list) if N_list else np.zeros((0, 3))
    NF = np.array(NF_list, dtype=int) if NF_list else np.zeros((0, ss), dtype=int)
    SI = np.array(SI_list) if SI_list else np.zeros((0, 2))

    return V, F, UV, TF, N, NF, SI
