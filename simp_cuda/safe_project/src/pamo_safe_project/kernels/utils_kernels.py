import warp as wp

@wp.func
def sym_eig3(A: wp.mat33):
    """
    Computes eigenvalues and eigenvectors for a symmetric 3x3 matrix 
    using a fixed-iteration Jacobi solver.
    Returns (eigenvalues, eigenvectors_matrix)
    """
    Q = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    d = wp.vec3(A[0, 0], A[1, 1], A[2, 2])
    
    # 3x3 symmetric matrix, we can run a fixed number of sweeps (e.g. 4)
    for _ in range(4):
        # --- (0, 1) ---
        off = A[0, 1]
        if wp.abs(off) > 1e-6:
            theta = 0.5 * (d[1] - d[0]) / off
            t = 1.0 / (wp.abs(theta) + wp.sqrt(theta*theta + 1.0))
            if theta < 0.0: t = -t
            c = 1.0 / wp.sqrt(t*t + 1.0)
            s = t * c
            tau = s / (1.0 + c)
            
            tmp = t * off
            d[0] = d[0] - tmp
            d[1] = d[1] + tmp
            A[0, 1] = 0.0
            
            for k in range(3):
                g = Q[k, 0]
                h = Q[k, 1]
                Q[k, 0] = g - s * (h + g * tau)
                Q[k, 1] = h + s * (g - h * tau)

        # --- (0, 2) ---
        off = A[0, 2]
        if wp.abs(off) > 1e-6:
            theta = 0.5 * (d[2] - d[0]) / off
            t = 1.0 / (wp.abs(theta) + wp.sqrt(theta*theta + 1.0))
            if theta < 0.0: t = -t
            c = 1.0 / wp.sqrt(t*t + 1.0)
            s = t * c
            tau = s / (1.0 + c)
            
            tmp = t * off
            d[0] = d[0] - tmp
            d[2] = d[2] + tmp
            A[0, 2] = 0.0
            
            for k in range(3):
                g = Q[k, 0]
                h = Q[k, 2]
                Q[k, 0] = g - s * (h + g * tau)
                Q[k, 2] = h + s * (g - h * tau)

        # --- (1, 2) ---
        off = A[1, 2]
        if wp.abs(off) > 1e-6:
            theta = 0.5 * (d[2] - d[1]) / off
            t = 1.0 / (wp.abs(theta) + wp.sqrt(theta*theta + 1.0))
            if theta < 0.0: t = -t
            c = 1.0 / wp.sqrt(t*t + 1.0)
            s = t * c
            tau = s / (1.0 + c)
            
            tmp = t * off
            d[1] = d[1] - tmp
            d[2] = d[2] + tmp
            A[1, 2] = 0.0
            
            for k in range(3):
                g = Q[k, 1]
                h = Q[k, 2]
                Q[k, 1] = g - s * (h + g * tau)
                Q[k, 2] = h + s * (g - h * tau)
                
    return d, Q

@wp.func
def spd_project_block(block: wp.mat33, it_max: int):
    # 1. Symmetrize
    sym = 0.5 * (block + wp.transpose(block))
    
    # 2. Eigendecomposition
    vals, vecs = sym_eig3(sym)
    
    # 3. Clamp Eigenvalues (Project to Positive Definite)
    v0 = wp.max(vals[0], 1e-4)
    v1 = wp.max(vals[1], 1e-4)
    v2 = wp.max(vals[2], 1e-4)
    
    # 4. Reconstruct: A = V * Clamp(D) * V^T
    D = wp.diag(wp.vec3(v0, v1, v2))
    return vecs * D * wp.transpose(vecs)

@wp.kernel
def a_plus_k_b_kernel(
    a: wp.array(dtype=wp.vec3),
    k: float,
    b: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ret[tid] = a[tid] + k * b[tid]

@wp.kernel
def block_spd_project_kernel(
    blocks: wp.array(dtype=wp.mat33, ndim=3), # UPDATED: ndim=3
    it_max: int,
):
    # UPDATED: Get 3D thread indices
    i, j, k = wp.tid()

    # UPDATED: Access 3D array
    val = blocks[i, j, k]

    projected_val = spd_project_block(val, it_max)

    # UPDATED: Write back to 3D array
    blocks[i, j, k] = projected_val