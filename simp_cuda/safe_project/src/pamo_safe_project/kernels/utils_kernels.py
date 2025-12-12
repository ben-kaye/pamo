import warp as wp

# It's best to pass epsilon as an argument to keep the function pure,
# but a global constant works if it's strictly typed.
EPSILON = 1e-4


@wp.func
def spd_project_block(block: wp.mat33, epsilon: float):
    """
    Projects a 3x3 block to be Symmetric Positive Definite.
    Reconstructs A_spd = Q * max(D, epsilon) * Q^T
    """
    # 1. Symmetrize
    sym = 0.5 * (block + wp.transpose(block))

    # 2. Eigendecomposition
    # The compiler error [mat33, vec3f] indicates your wp.eig3 returns (mat33, vec3).
    # We unpack into Q (eigenvectors matrix) and w (eigenvalues vector).
    Q, w = wp.eig3(sym)

    # 3. Vectorized Clamp
    # Create a vector of epsilons for SIMD-friendly max
    eps_vec = wp.vec3(epsilon, epsilon, epsilon)

    # w is vec3, eps_vec is vec3. This matches the [vec3, vec3] overload.
    w_clamped = wp.max(w, eps_vec)

    # 4. Reconstruct
    # D = diag(w_clamped)
    D = wp.diag(w_clamped)

    # Q * D * Q^T
    # Warp executes matrix multiplications left-to-right.
    # (Q * D) scales columns of Q by eigenvalues.
    return Q * D * wp.transpose(Q)


@wp.kernel
def block_spd_project_kernel(
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    it_max: int,
):
    i, j, k = wp.tid()

    # Load the matrix
    val = blocks[i, j, k]

    # Project
    projected_val = spd_project_block(val, EPSILON)

    # Store result
    blocks[i, j, k] = projected_val
