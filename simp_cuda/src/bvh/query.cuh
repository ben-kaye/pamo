#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH
#define STACK_SIZE 256
// Sentinel when the fixed depth-first stack would overflow. Callers must
// hard-fail: a truncated walk can miss real intersections.
#define LBVH_STACK_OVERFLOW 0xFFFFFFFFu
#include "predicator.cuh"

namespace lbvh
{
    // Fixed-stack depth-first walk over the LBVH for AABB-overlap queries.
    //
    // Node encoding: object_idx != 0xFFFFFFFF means leaf (object = face index);
    // otherwise left_idx/right_idx are internal children. Root is always node 0.
    //
    // face_i_raw[obj * face_i_stride] == -1 => deleted face (skip as partner).
    // For packed Triangle<int>{i,j,k}, pass (int*)F and face_i_stride=3 so the
    // first component of each triangle is the deleted-marker.
    //
    // max_candidates caps how many partner leaves are counted/written.
    // Returns LBVH_STACK_OVERFLOW if the stack would exceed STACK_SIZE.
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
        *stack_ptr++ = 0; // root

        unsigned int num_found = 0;
        do
        {
            // Need room for both children; fail rather than drop a subtree.
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
                else // internal: push child for later visit
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
                else
                {
                    *stack_ptr++ = R_idx;
                }
            }

        } while (stack < stack_ptr);
        return num_found;
    }

    // Same depth-first walk as get_number_of_intersect_candidates, but writes
    // each accepted partner as (query_idx, obj_idx) into outiter starting at
    // index `first` (two slots per pair). Stops at max_candidates pairs.
    // On stack overflow, partial writes may already exist — host must not trust
    // the list and should treat the return as failure.
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