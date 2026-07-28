#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH
#define STACK_SIZE 256
// Sentinel returned when the fixed DFS stack would overflow. Callers must treat
// this as a hard failure: a partial candidate set can miss real intersections,
// and the per-face max_candidates cap cannot detect that truncation.
#define LBVH_STACK_OVERFLOW 0xFFFFFFFFu
#include "predicator.cuh"

namespace lbvh
{
    // face_i_raw[obj * face_i_stride] == -1 means deleted face (skip as partner).
    // For packed Triangle<int> {i,j,k}, pass (int*)F and face_i_stride=3.
    // max_candidates caps pairs counted/written per query.
    // Returns LBVH_STACK_OVERFLOW if the fixed traversal stack fills.
    template <typename Real, typename Objects, bool IsConst>
    __device__ unsigned int get_number_of_intersect_candidates(
        const detail::basic_device_bvh<Real, Objects, IsConst> &bvh,
        const query_overlap<Real> q,
        const unsigned int query_idx,
        const int *face_i_raw = nullptr,
        const unsigned int face_i_stride = 1,
        const unsigned int max_candidates = 0xFFFFFFFF) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;

        index_type stack[STACK_SIZE];
        index_type *stack_ptr = stack;
        *stack_ptr++ = 0; // root node is always 0

        unsigned int num_found = 0;
        do
        {
            // Fixed stack full: fail loudly rather than silently truncating.
            if (stack_ptr - stack >= STACK_SIZE - 2)
                return LBVH_STACK_OVERFLOW;

            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            if (intersects(q.target, bvh.aabbs[L_idx]))
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    const bool deleted = face_i_raw != nullptr &&
                        face_i_raw[obj_idx * face_i_stride] == -1;
                    if (obj_idx != query_idx && !deleted && num_found < max_candidates)
                    {
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = L_idx;
                }
            }
            if (intersects(q.target, bvh.aabbs[R_idx]))
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    const bool deleted = face_i_raw != nullptr &&
                        face_i_raw[obj_idx * face_i_stride] == -1;
                    if (obj_idx != query_idx && !deleted && num_found < max_candidates)
                    {
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = R_idx;
                }
            }

        } while (stack < stack_ptr);
        return num_found;
    }



    // dfs code ----------------------
    // Writes (query_idx, obj_idx) pairs into outiter starting at index `first`.
    // Each pair occupies 2 slots. Stops at max_candidates pairs.
    // Returns LBVH_STACK_OVERFLOW if the fixed traversal stack fills (partial
    // writes may have already occurred; host must not trust the list).
    template <typename Real, typename Objects, bool IsConst, typename OutputIterator>
    __device__ unsigned int query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst> &bvh,
        const query_overlap<Real> q, OutputIterator outiter,
        const unsigned int query_idx,
        const unsigned int first = 0,
        const int *face_i_raw = nullptr,
        const unsigned int face_i_stride = 1,
        const unsigned int max_candidates = 0xFFFFFFFF) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;

        index_type stack[STACK_SIZE];
        index_type *stack_ptr = stack;
        *stack_ptr++ = 0; // root node is always 0
        unsigned int num_found = 0;

        // dynamic buffer
        outiter += first;
        do
        {
            if (stack_ptr - stack >= STACK_SIZE - 2)
                return LBVH_STACK_OVERFLOW;

            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            if (intersects(q.target, bvh.aabbs[L_idx]))
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    const bool deleted = face_i_raw != nullptr &&
                        face_i_raw[obj_idx * face_i_stride] == -1;
                    if (obj_idx != query_idx && !deleted && num_found < max_candidates)
                    {
                        *outiter++ = query_idx;
                        *outiter++ = obj_idx;
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = L_idx;
                }
            }
            if (intersects(q.target, bvh.aabbs[R_idx]))
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    const bool deleted = face_i_raw != nullptr &&
                        face_i_raw[obj_idx * face_i_stride] == -1;
                    if (obj_idx != query_idx && !deleted && num_found < max_candidates)
                    {
                        *outiter++ = query_idx;
                        *outiter++ = obj_idx;
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = R_idx;
                }
            }
        } while (stack < stack_ptr);
        return num_found;
    }

}

#endif // LBVH_QUERY_CUH