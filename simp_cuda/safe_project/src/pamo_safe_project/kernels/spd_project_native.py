"""PSD projection via Warp @wp.func_native — replaces the Rabbit-Hu Warp fork builtin.

The hinge Hessian path needs ``spd_project_blocks`` (Jacobi eigen-decomposition of a
symmetric matrix stored as 3x3 mat33 blocks).  That was the only real reason for the
forked Warp; this module ships the same algorithm as a native snippet on stock
``warp-lang``.

Source algorithm: Burkardt Jacobi eigenvalue (LGPL), as adapted in Rabbit-Hu/warp
``warp/native/spd_project.h``.
"""

from __future__ import annotations

import warp as wp

# Single self-contained snippet: helpers are static-inline C++ functions at the
# top of the generated translation unit... except func_native only injects a
# *function body*.  So we put everything in one body with no nested defs, using
# a fixed 9x9 workspace (n_blocks <= 3).
_SPD_PROJECT_BLOCKS_SNIPPET = r"""
    // Project the top-left n x n block-matrix of 3x3 tiles (n <= 3) to PSD.
    // b is a 2D array of mat33; only b[0:n, 0:n] is modified.
    const int n_max = 3;
    const int dim_max = 9;  // n_max * 3
    if (n < 1 || n > n_max) {
        return;
    }

    const int dim = n * 3;
    const int b_size = (int)b.shape[1];
    float a[81];
    float v[81];
    float d[9];
    float bw[9];
    float zw[9];

    // blocks_to_array: pack mat33 tiles into a contiguous dim x dim matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // mat33 stored as row-major 3x3 inside warp::mat_t
            auto& tile = b.data[i * b_size + j];
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    a[(i * 3 + k) * dim + (j * 3 + l)] = tile.data[k][l];
                }
            }
        }
    }

    // ---- jacobi_eigenvalue (Burkardt) on a[0:dim, 0:dim] ----
    // identity -> v
    for (int j = 0; j < dim; ++j) {
        for (int i = 0; i < dim; ++i) {
            v[i + j * dim] = (i == j) ? 1.0f : 0.0f;
        }
    }
    // diagonal of a -> d
    for (int i = 0; i < dim; ++i) {
        d[i] = a[i + i * dim];
        bw[i] = d[i];
        zw[i] = 0.0f;
    }

    int it_num = 0;
    while (it_num < it_max) {
        ++it_num;

        float thresh = 0.0f;
        for (int j = 0; j < dim; ++j) {
            for (int i = 0; i < j; ++i) {
                float aij = a[i + j * dim];
                thresh += aij * aij;
            }
        }
        thresh = sqrtf(thresh) / (float)(4 * dim);
        if (thresh == 0.0f) {
            break;
        }

        for (int p = 0; p < dim; ++p) {
            for (int q = p + 1; q < dim; ++q) {
                float apq = a[p + q * dim];
                float gapq = 10.0f * fabsf(apq);
                float termp = gapq + fabsf(d[p]);
                float termq = gapq + fabsf(d[q]);

                if (4 < it_num && termp == fabsf(d[p]) && termq == fabsf(d[q])) {
                    a[p + q * dim] = 0.0f;
                } else if (thresh <= fabsf(apq)) {
                    float h = d[q] - d[p];
                    float term = fabsf(h) + gapq;
                    float t;
                    if (term == fabsf(h)) {
                        t = apq / h;
                    } else {
                        float theta = 0.5f * h / apq;
                        t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta * theta));
                        if (theta < 0.0f) {
                            t = -t;
                        }
                    }
                    float c = 1.0f / sqrtf(1.0f + t * t);
                    float s = t * c;
                    float tau = s / (1.0f + c);
                    h = t * apq;

                    zw[p] -= h;
                    zw[q] += h;
                    d[p] -= h;
                    d[q] += h;
                    a[p + q * dim] = 0.0f;

                    for (int j = 0; j < p; ++j) {
                        float g = a[j + p * dim];
                        float hh = a[j + q * dim];
                        a[j + p * dim] = g - s * (hh + g * tau);
                        a[j + q * dim] = hh + s * (g - hh * tau);
                    }
                    for (int j = p + 1; j < q; ++j) {
                        float g = a[p + j * dim];
                        float hh = a[j + q * dim];
                        a[p + j * dim] = g - s * (hh + g * tau);
                        a[j + q * dim] = hh + s * (g - hh * tau);
                    }
                    for (int j = q + 1; j < dim; ++j) {
                        float g = a[p + j * dim];
                        float hh = a[q + j * dim];
                        a[p + j * dim] = g - s * (hh + g * tau);
                        a[q + j * dim] = hh + s * (g - hh * tau);
                    }
                    for (int j = 0; j < dim; ++j) {
                        float g = v[j + p * dim];
                        float hh = v[j + q * dim];
                        v[j + p * dim] = g - s * (hh + g * tau);
                        v[j + q * dim] = hh + s * (g - hh * tau);
                    }
                }
            }
        }

        for (int i = 0; i < dim; ++i) {
            bw[i] += zw[i];
            d[i] = bw[i];
            zw[i] = 0.0f;
        }
    }

    // restore upper triangle of a (not needed further, but matches reference)
    for (int j = 0; j < dim; ++j) {
        for (int i = 0; i < j; ++i) {
            a[i + j * dim] = a[j + i * dim];
        }
    }

    // ascending sort eigenvalues / eigenvectors
    for (int k = 0; k < dim - 1; ++k) {
        int m = k;
        for (int l = k + 1; l < dim; ++l) {
            if (d[l] < d[m]) {
                m = l;
            }
        }
        if (m != k) {
            float t = d[m];
            d[m] = d[k];
            d[k] = t;
            for (int i = 0; i < dim; ++i) {
                float w = v[i + m * dim];
                v[i + m * dim] = v[i + k * dim];
                v[i + k * dim] = w;
            }
        }
    }

    // zero a, then a = V * max(D,0) * V^T
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            a[i * dim + j] = 0.0f;
        }
    }
    for (int k = 0; k < dim; ++k) {
        if (d[k] > 0.0f) {
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    a[i * dim + j] += d[k] * v[k * dim + i] * v[k * dim + j];
                }
            }
        }
    }

    // array_to_blocks: unpack, re-symmetrize
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            auto& tile = b.data[i * b_size + j];
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    int i_a = i * 3 + k;
                    int j_a = j * 3 + l;
                    tile.data[k][l] = (a[i_a * dim + j_a] + a[j_a * dim + i_a]) * 0.5f;
                }
            }
        }
    }
"""


@wp.func_native(_SPD_PROJECT_BLOCKS_SNIPPET)
def spd_project_blocks(
    n: int,
    b: wp.array(dtype=wp.mat33, ndim=2),
    it_max: int,
):
    """Project an n×n block matrix of mat33 tiles (n≤3) to PSD, in place."""
    ...
