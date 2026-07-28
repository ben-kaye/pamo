import warp as wp


@wp.kernel
def update_p_r_z_compute_zr_kernel(
    v: wp.array(dtype=wp.vec3),
    A_v: wp.array(dtype=wp.vec3),
    v_A_v: wp.array(dtype=float),
    zr: wp.array(dtype=float),
    diag: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
    zr_new: wp.array(dtype=float),
):
    tid = wp.tid()

    # Guard breakdown: skip state update when p^T A p is non-positive or
    # residual inner product is non-finite / already solved.
    pap = v_A_v[0]
    zr0 = zr[0]
    if pap > 1e-16 and wp.abs(zr0) > 1e-30:
        alpha = zr0 / pap
        p[tid] = p[tid] + alpha * v[tid]
        r[tid] = r[tid] - alpha * A_v[tid]
    # Jacobi preconditioner with epsilon already applied elsewhere; still
    # protect against exact-zero diagonal components.
    z[tid] = wp.cw_div(r[tid], diag[tid] + wp.vec3(1e-6))
    wp.atomic_add(zr_new, 0, wp.dot(z[tid], r[tid]))


@wp.kernel
def update_v_kernel(
    z: wp.array(dtype=wp.vec3),
    zr: wp.array(dtype=float),
    zr_new: wp.array(dtype=float),
    v: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # Guard zr_new/zr: zero residual, underflow, or NaN must not produce Inf/NaN.
    zr0 = zr[0]
    zr1 = zr_new[0]
    if wp.abs(zr0) > 1e-30 and zr0 == zr0 and zr1 == zr1:
        s = zr1 / zr0
        v[tid] = z[tid] + s * v[tid]
    else:
        # Breakdown or converged: restart search direction as preconditioned residual.
        v[tid] = z[tid]


@wp.kernel
def compute_dot_kernel(
    x: wp.array(dtype=wp.vec3),
    y: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=float),
):
    tid = wp.tid()

    wp.atomic_add(ret, 0, wp.dot(x[tid], y[tid]))
    

@wp.kernel
def compute_block_diag_inv_kernel(
    diag: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
):
    # z = r / diag
    tid = wp.tid()
    z[tid] = wp.cw_div(r[tid], diag[tid] + wp.vec3(1e-6))


@wp.kernel
def line_search_kernel(
    q: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=float),
    n_halves: float,
    energy_prev: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    q_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    # Keep an accepted (strictly decreasing, finite) candidate.
    e = energy[0]
    e_prev = energy_prev[0]
    if n_halves > 0.0 and e == e and e < e_prev:
        return
    q_new[tid] = q[tid] + alpha[0] * wp.pow(0.5, n_halves) * p[tid]


@wp.kernel
def restore_q_kernel(
    q_prev: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    q[tid] = q_prev[tid]


@wp.kernel
def clamp_p_kernel(
    p: wp.array(dtype=wp.vec3),
    q_prev_newton: wp.array(dtype=wp.vec3),
    q_prev_detection: wp.array(dtype=wp.vec3),
    radius: float,
):
    i = wp.tid()
    p_i = p[i]
    q_n_i = q_prev_newton[i]
    q_d_i = q_prev_detection[i]
    # Wanna go to q = q_n_i + p_i
    # but requires |q - q_d_i| <= radius
    delta_q = q_n_i + p_i - q_d_i
    
    if wp.length(delta_q) > radius:
        p_i_new = wp.normalize(delta_q) * radius + q_d_i - q_n_i
        p[i] = p_i_new
