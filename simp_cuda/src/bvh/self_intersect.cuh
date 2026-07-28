#ifndef LBVH_SELF_INTERSECT_CUH
#define LBVH_SELF_INTERSECT_CUH
#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "bvh.cuh"
#include "query.cuh"
#include "types.cuh"
#include "predicator.cuh"
#include "tri_tri_3d.cuh"
#include "tri_tri_2d.cuh"

// Self-intersection pipeline (broad-phase then narrow-phase):
//
//   1. Build per-face geometry for LBVH (deleted faces -> tiny placeholders).
//   2. Broad phase, two-pass packing so every AABB-overlap pair is listed:
//        first pass: per face, count overlapping leaves (uncapped)
//        exclusive_scan -> write offsets (first_query_result)
//        grow intersect_candidates to total_slots
//        second pass: re-walk BVH, write (query, partner) pairs at those offsets
//      Each pair is two unsigned integers; total_slots = 2 * candidate pair count.
//   3. Narrow phase: exact triangle-triangle test on each candidate; skip
//      topological neighbors (shared edge / vertex-at-epsilon) so manifold
//      adjacency is not treated as a self-intersection.
//   4. Record intersecting face indices for the collapse-undo consumer.
//
// Soft resource limit refuses pathological AABB density rather than running
// out of memory.
#ifndef SELF_X_MAX_TOTAL_SLOTS
#define SELF_X_MAX_TOTAL_SLOTS (1u << 28)  // ~1 GiB of unsigned ints for candidates
#endif

using namespace std;

namespace cusimp_free {
    class CUSimp_Free;
}

namespace selfx{
    const int BLOCK_SIZE = 512;

    // Host-side helper: throw on CUDA API failure (matches stage-2 policy).
    inline void check_cuda(cudaError_t code, const char *file, int line)
    {
        if (code == cudaSuccess)
            return;
        fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code),
                cudaGetErrorString(code), file, line);
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(code));
    }
#define SELF_X_CHECK(code) selfx::check_cuda((code), __FILE__, __LINE__)

    __device__ __host__
    inline bool are_vertices_same(const float3 v1, const float3 v2, float epsilon){
        return std::abs(v1.x - v2.x) < epsilon &&
           std::abs(v1.y - v2.y) < epsilon &&
           std::abs(v1.z - v2.z) < epsilon;
    }

    __device__ __host__
    inline bool are_vertices_same(const float3 v1, const float3 v2){
        return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
    }

    // if triangle pair has shared edge
    __device__ __host__
    inline bool detect_shared_edge_coord(const float3 p1, const float3 q1, const float3 r1, 
                            const float3 p2, const float3 q2, const float3 r2){
        float epsilon = 1e-6;
        int shared_vertices = 0;
        shared_vertices += are_vertices_same(p1, p2, epsilon) || are_vertices_same(p1, q2, epsilon) || are_vertices_same(p1, r2, epsilon);
        shared_vertices += are_vertices_same(q1, p2, epsilon) || are_vertices_same(q1, q2, epsilon) || are_vertices_same(q1, r2, epsilon);
        shared_vertices += are_vertices_same(r1, p2, epsilon) || are_vertices_same(r1, q2, epsilon) || are_vertices_same(r1, r2, epsilon);

        return shared_vertices >= 2;
    }

    // Pass 1: for each live face, count BVH leaves whose AABBs overlap this face.
    // Output is slot counts (2 * #pairs) for exclusive_scan layout of pass 2.
    // F is Triangle<int>{i,j,k}; face.i == -1 marks deleted. BVH query sees F as
    // int* with stride 3 so partner leaves with i==-1 are skipped.
    __global__ void compute_num_of_query_result_kernel(
        cusimp_free::Triangle<int>* F_d_raw,
        lbvh::bvh_device<float, selfx::Triangle<float3>> bvh_dev,
        unsigned int* num_found_query_raw,
        std::size_t num_faces)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= (int)num_faces) return;
        if (F_d_raw[idx].i == -1) {
            num_found_query_raw[idx] = 0;
            return;
        }

        const auto self = bvh_dev.objects[idx];
        lbvh::aabb<float> query_box;
        float minX = fminf(self.v0.x, fminf(self.v1.x, self.v2.x));
        float minY = fminf(self.v0.y, fminf(self.v1.y, self.v2.y));
        float minZ = fminf(self.v0.z, fminf(self.v1.z, self.v2.z));
        float maxX = fmaxf(self.v0.x, fmaxf(self.v1.x, self.v2.x));
        float maxY = fmaxf(self.v0.y, fmaxf(self.v1.y, self.v2.y));
        float maxZ = fmaxf(self.v0.z, fmaxf(self.v1.z, self.v2.z));
        query_box.lower = make_float4(minX, minY, minZ, 0);
        query_box.upper = make_float4(maxX, maxY, maxZ, 0);

        const int *face_i_raw = reinterpret_cast<const int *>(F_d_raw);
        unsigned int num_found = lbvh::get_number_of_intersect_candidates(
            bvh_dev, lbvh::overlaps(query_box), (unsigned int)idx,
            face_i_raw, /*stride=*/3u);
        // LBVH_STACK_OVERFLOW and 2*count wrap both become this sentinel.
        // Host must hard-fail before exclusive_scan (partial counts miss
        // self-intersections).
        if (num_found == LBVH_STACK_OVERFLOW || num_found > 0x7FFFFFFFu) {
            num_found_query_raw[idx] = LBVH_STACK_OVERFLOW;
            return;
        }
        // Store slot count (two unsigned ints per candidate pair) for exclusive_scan.
        num_found_query_raw[idx] = 2u * num_found;
    }

    // Pass 2: re-walk BVH and pack (query_idx, partner_idx) at first_query_result[idx].
    // max_pairs is the count-pass budget so a divergent second walk cannot
    // write past the exclusive_scan layout.
    __global__ void compute_query_list_kernel(
        cusimp_free::Triangle<int>* F_d_raw,
        lbvh::bvh_device<float, selfx::Triangle<float3>> bvh_dev,
        unsigned int* first_query_result_raw,
        unsigned int* num_found_query_raw,
        unsigned int* intersect_candidates_raw,
        std::size_t num_faces)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= (int)num_faces) return;
        if (F_d_raw[idx].i == -1) return;

        const auto self = bvh_dev.objects[idx];
        lbvh::aabb<float> query_box;
        float minX = fminf(self.v0.x, fminf(self.v1.x, self.v2.x));
        float minY = fminf(self.v0.y, fminf(self.v1.y, self.v2.y));
        float minZ = fminf(self.v0.z, fminf(self.v1.z, self.v2.z));
        float maxX = fmaxf(self.v0.x, fmaxf(self.v1.x, self.v2.x));
        float maxY = fmaxf(self.v0.y, fmaxf(self.v1.y, self.v2.y));
        float maxZ = fmaxf(self.v0.z, fmaxf(self.v1.z, self.v2.z));
        query_box.lower = make_float4(minX, minY, minZ, 0);
        query_box.upper = make_float4(maxX, maxY, maxZ, 0);

        const int *face_i_raw = reinterpret_cast<const int *>(F_d_raw);
        unsigned int first = first_query_result_raw[idx];
        unsigned int slots = num_found_query_raw[idx];
        unsigned int max_pairs = slots / 2u;
        // Count pass already hard-fails on stack overflow; ignore return here.
        (void)lbvh::query_device(bvh_dev, lbvh::overlaps(query_box),
                                 intersect_candidates_raw, (unsigned int)idx, first,
                                 face_i_raw, /*stride=*/3u, max_pairs);
    }

    void ensure_bvh_storage_size(cusimp_free::CUSimp_Free *sp)
    {
        // resize (not reserve): kernels write [0, num_faces); exclusive_scan
        // needs a +1 past-the-end slot for the total. Candidate buffer is grown
        // after the count pass from the scan total.
        sp->bvh_triangles.resize(sp->allocated_tris);
        sp->num_found_query.resize(sp->allocated_tris + 1);
        sp->first_query_result.resize(sp->allocated_tris + 1);
    }

    // Grow-only self-intersect result scratch on CUSimp_Free
    // (no per-call malloc/free). capacity_ints is the max number of face-index
    // slots we will store.
    void ensure_si_scratch(cusimp_free::CUSimp_Free *sp, unsigned int capacity_ints)
    {
        if (sp->si_is_intersect == nullptr) {
            SELF_X_CHECK(cudaMalloc((void **)&sp->si_is_intersect, sizeof(unsigned int)));
        }
        if (sp->si_total == nullptr) {
            SELF_X_CHECK(cudaMalloc((void **)&sp->si_total, sizeof(unsigned int)));
        }
        if (sp->si_stored == nullptr) {
            SELF_X_CHECK(cudaMalloc((void **)&sp->si_stored, sizeof(unsigned int)));
        }
        if (capacity_ints > sp->allocated_si_intersections) {
            size_t new_cap = (size_t)capacity_ints + (size_t)capacity_ints / 5u + 1u;
            SELF_X_CHECK(cudaFree(sp->si_intersections));
            sp->si_intersections = nullptr;
            if (new_cap > 0) {
                SELF_X_CHECK(cudaMalloc((void **)&sp->si_intersections,
                                        new_cap * sizeof(unsigned int)));
            }
            sp->allocated_si_intersections = new_cap;
        }
    }

    // Integer-center shift: subtract floor(mean) of the six triangle vertices so
    // triangle-triangle predicates run near the origin and stay better conditioned.
    __device__
    inline int floor_mean(float v1, float v2, float v3, float v4, float v5, float v6) {
        float mean = (v1 + v2 + v3 + v4 + v5 + v6) / 6.0f;
        return (int)floorf(mean);
    }

    __device__
    void translate_coordinates(float3 *p1, float3 *q1, float3 *r1, float3 *p2, float3 *q2, float3 *r2) {
        int mean_x = floor_mean(p1->x, q1->x, r1->x, p2->x, q2->x, r2->x);
        int mean_y = floor_mean(p1->y, q1->y, r1->y, p2->y, q2->y, r2->y);
        int mean_z = floor_mean(p1->z, q1->z, r1->z, p2->z, q2->z, r2->z);

        p1->x -= mean_x; p1->y -= mean_y; p1->z -= mean_z;
        q1->x -= mean_x; q1->y -= mean_y; q1->z -= mean_z;
        r1->x -= mean_x; r1->y -= mean_y; r1->z -= mean_z;
        p2->x -= mean_x; p2->y -= mean_y; p2->z -= mean_z;
        q2->x -= mean_x; q2->y -= mean_y; q2->z -= mean_z;
        r2->x -= mean_x; r2->y -= mean_y; r2->z -= mean_z;
    }

    // Returns true if any non-adjacent face pair properly intersects.
    // Side effect: fills sp->intersected_triangle_idx / n_intersect with the
    // face indices that participate in intersections (for collapse undo).
    bool self_intersect(cusimp_free::CUSimp_Free *sp, unsigned int num_vertices, unsigned int num_faces, float epsilon) {
        if (num_faces == 0)
            return false;

        cusimp_free::Vertex<float>* V_d_raw = sp->points;
        cusimp_free::Triangle<int>* F_d_raw = sp->triangles;
        Triangle<float3>* triangles_d_raw = thrust::raw_pointer_cast(sp->bvh_triangles.data());

        // get triangle data to build bvh -----------------
        // Removed faces are marked with i == -1 (see remove_invalid_faces).
        // Must not index V_d_raw[-1] (out of bounds). Placeholders use a tiny
        // non-zero triangle so they don't all share the origin AABB; query
        // kernels also skip i==-1 as query and partner.
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<std::size_t>(0),
                         thrust::make_counting_iterator<std::size_t>(num_faces),
                         [V_d_raw, F_d_raw, triangles_d_raw] __device__(std::size_t idx){
                            Triangle<float3> tri;
                            int v0_row = F_d_raw[idx].i;
                            int v1_row = F_d_raw[idx].j;
                            int v2_row = F_d_raw[idx].k;
                            if (v0_row < 0 || v1_row < 0 || v2_row < 0) {
                                // Degenerate placeholder; query kernels skip i==-1 faces.
                                // Jitter by idx so deleted leaves don't all share one AABB.
                                float s = 1e-20f * (float)(idx + 1);
                                tri.v0 = make_float3(s, 0.f, 0.f);
                                tri.v1 = make_float3(0.f, s, 0.f);
                                tri.v2 = make_float3(0.f, 0.f, s);
                                triangles_d_raw[idx] = tri;
                                return;
                            }
                            tri.v0 = make_float3(V_d_raw[v0_row].x, V_d_raw[v0_row].y, V_d_raw[v0_row].z);
                            tri.v1 = make_float3(V_d_raw[v1_row].x, V_d_raw[v1_row].y, V_d_raw[v1_row].z);
                            tri.v2 = make_float3(V_d_raw[v2_row].x, V_d_raw[v2_row].y, V_d_raw[v2_row].z);
                            triangles_d_raw[idx] = tri;
                         });

        // construct bvh -------------------------------

        lbvh::bvh<float, selfx::Triangle<float3>, aabb_getter> bvh(sp->bvh_triangles.begin(), sp->bvh_triangles.begin() + num_faces, false);
        // get device ptr
        const auto bvh_dev = bvh.get_device_repr();

        // run query ----------------------------
        thrust::fill(thrust::device, sp->num_found_query.begin(), sp->num_found_query.end(), 0);

        // get raw pointer
        unsigned int* num_found_results_raw = thrust::raw_pointer_cast(sp->num_found_query.data());
        unsigned int* first_query_result_raw = thrust::raw_pointer_cast(sp->first_query_result.data());

        const int n_blocks = (int)((num_faces + BLOCK_SIZE - 1) / BLOCK_SIZE);
        // Pass 1: count AABB-overlap candidates per face (uncapped).
        compute_num_of_query_result_kernel<<<n_blocks, BLOCK_SIZE>>>(
            F_d_raw, bvh_dev, num_found_results_raw, num_faces);
        SELF_X_CHECK(cudaGetLastError());
        SELF_X_CHECK(cudaDeviceSynchronize());

        // Fail before exclusive_scan: LBVH_STACK_OVERFLOW is 0xFFFFFFFF and would
        // poison the prefix sum. Truncated walks miss real intersections.
        {
            thrust::device_ptr<unsigned int> num_found_ptr(num_found_results_raw);
            unsigned int max_slots = thrust::reduce(
                thrust::device, num_found_ptr, num_found_ptr + num_faces, 0u,
                thrust::maximum<unsigned int>());
            if (max_slots == LBVH_STACK_OVERFLOW) {
                throw std::runtime_error(
                    "self_intersect: BVH traversal stack overflow (STACK_SIZE=" +
                    std::to_string(STACK_SIZE) +
                    "); candidate set may be incomplete. "
                    "Intersection guarantee cannot be maintained.");
            }
        }

        thrust::exclusive_scan(thrust::device, num_found_results_raw,
                               num_found_results_raw + num_faces + 1,
                               sp->first_query_result.data());

        // total packed slots (= 2 * total candidate pairs)
        unsigned int total_slots = 0;
        SELF_X_CHECK(cudaMemcpy(&total_slots, first_query_result_raw + num_faces,
                                sizeof(unsigned int), cudaMemcpyDeviceToHost));

        if (total_slots > (unsigned int)SELF_X_MAX_TOTAL_SLOTS) {
            throw std::runtime_error(
                "self_intersect: candidate packing total_slots=" +
                std::to_string(total_slots) + " exceeds resource limit " +
                std::to_string((unsigned int)SELF_X_MAX_TOTAL_SLOTS) +
                " (pathological AABB density).");
        }

        // Pass 2 prep: grow packed candidate buffer to exclusive_scan total.
        if ((size_t)total_slots > sp->intersect_candidates.size()) {
            sp->intersect_candidates.resize((size_t)total_slots + (size_t)total_slots / 5u + 1u);
        }
        if (total_slots > 0) {
            thrust::fill(thrust::device, sp->intersect_candidates.begin(),
                         sp->intersect_candidates.begin() + total_slots, 0xFFFFFFFFu);
        }
        unsigned int* intersect_candidates_raw =
            thrust::raw_pointer_cast(sp->intersect_candidates.data());
        first_query_result_raw = thrust::raw_pointer_cast(sp->first_query_result.data());
        num_found_results_raw = thrust::raw_pointer_cast(sp->num_found_query.data());

        // Pass 2: pack (query, partner) pairs into the scanned layout.
        if (total_slots > 0) {
            compute_query_list_kernel<<<n_blocks, BLOCK_SIZE>>>(
                F_d_raw, bvh_dev, first_query_result_raw, num_found_results_raw,
                intersect_candidates_raw, num_faces);
            SELF_X_CHECK(cudaGetLastError());
            SELF_X_CHECK(cudaDeviceSynchronize());
        }

        // Narrow phase: exact triangle-triangle test on each broad-phase pair.
        // Result buffer holds face indices (two per intersecting pair); capacity
        // is 2*F so each face can appear at most once as a stored index pair.
        const unsigned int capacity_ints = 2u * (unsigned int)num_faces;
        ensure_si_scratch(sp, capacity_ints);

        unsigned int* d_isIntersect = sp->si_is_intersect;
        unsigned int* d_intersections = sp->si_intersections;
        unsigned int* d_total = sp->si_total;   // attempted writes (overflow detect)
        unsigned int* d_stored = sp->si_stored; // successful writes into d_intersections

        unsigned int h_isIntersect = 0;
        SELF_X_CHECK(cudaMemcpy(d_isIntersect, &h_isIntersect, sizeof(unsigned int), cudaMemcpyHostToDevice));
        SELF_X_CHECK(cudaMemset(d_total, 0, sizeof(unsigned int)));
        SELF_X_CHECK(cudaMemset(d_stored, 0, sizeof(unsigned int)));
        if (capacity_ints > 0 && d_intersections != nullptr) {
            SELF_X_CHECK(cudaMemset(d_intersections, 0xFF,
                                    (size_t)capacity_ints * sizeof(unsigned int)));
        }

        unsigned int num_query_result = total_slots / 2u;

        if (num_query_result > 0) {
            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator<unsigned int>(0),
                             thrust::make_counting_iterator<unsigned int>(num_query_result),
                             [epsilon, d_isIntersect, d_total, d_stored, d_intersections, capacity_ints, triangles_d_raw, intersect_candidates_raw, F_d_raw] __device__(std::size_t idx) {
                                unsigned int query_idx = intersect_candidates_raw[2 * idx];
                                unsigned int current_idx = intersect_candidates_raw[2 * idx + 1];

                                if(query_idx == 0xFFFFFFFF) return;
                                if(current_idx == 0xFFFFFFFF) return;
                                if(F_d_raw[query_idx].i == -1) return;
                                if(F_d_raw[current_idx].i == -1) return;

                                cusimp_free::Triangle<int> current_face = F_d_raw[current_idx];
                                cusimp_free::Triangle<int> query_face = F_d_raw[query_idx];

                                Triangle<float3> current_tris = triangles_d_raw[current_idx];
                                Triangle<float3> query_tris = triangles_d_raw[query_idx];

                                int vertices_current[] = {current_face.i, current_face.j, current_face.k};
                                int vertices_query[] = {query_face.i, query_face.j, query_face.k};
                                int num_count = 0; // shared vertex *indices* (topology)

                                float3 p1,q1,r1,p2,q2,r2;
                                p1 = current_tris.v0;
                                q1 = current_tris.v1;
                                r1 = current_tris.v2;
                                p2 = query_tris.v0;
                                q2 = query_tris.v1;
                                r2 = query_tris.v2;

                                translate_coordinates(&p1, &q1, &r1, &p2, &q2, &r2);

                                for(unsigned int j = 0; j < 3; j++){
                                    int vertex_current = vertices_current[j];
                                    for(unsigned int k = 0; k < 3; k++){
                                        if(vertex_current == vertices_query[k]){
                                            num_count++;
                                        }
                                    }
                                }
                                float tri_a[3][3];
                                float tri_b[3][3];
                                copy_v3_v3_float_float3(tri_a[0], p1);
                                copy_v3_v3_float_float3(tri_a[1], q1);
                                copy_v3_v3_float_float3(tri_a[2], r1);
                                copy_v3_v3_float_float3(tri_b[0], p2);
                                copy_v3_v3_float_float3(tri_b[1], q2);
                                copy_v3_v3_float_float3(tri_b[2], r2);

                                // Filters (not true self-intersections for our mesh):
                                //  - coplanar: no implemented coplanar test; skip
                                //  - num_count==2 or coord-shared edge: manifold neighbors
                                //  - shared vertex + tiny intersection segment: contact at vertex
                                if(is_coplanar(tri_a, tri_b)){
                                    return;
                                }

                                if(num_count == 2){
                                    return;
                                }
                                else if(detect_shared_edge_coord(p1,q1,r1,p2,q2,r2)){
                                    return;
                                }

                                float3 source, target;
                                source = make_float3(1,1,1);
                                target = make_float3(-1,-1,-1);

                                float r_i1[3];
                                float r_i2[3];
                                bool isIntersecting = isect_tri_tri_v3(p1,q1,r1,p2,q2,r2,r_i1,r_i2);

                                if(isIntersecting){
                                    copy_v3_v3_float3_float(source, r_i1);
                                    copy_v3_v3_float3_float(target, r_i2);
                                    float dist = largest_distance(source, target);
                                    bool sharedVertex = (num_count == 1);
                                    if(dist < epsilon && sharedVertex){
                                        return;
                                    }
                                    atomicExch(d_isIntersect, 1u);
                                    // Always count into d_total; only write if capacity remains.
                                    unsigned int pos = atomicAdd(d_total, 2u);
                                    if (d_intersections != nullptr && pos + 1u < capacity_ints) {
                                        d_intersections[pos] = query_idx;
                                        d_intersections[pos + 1u] = current_idx;
                                        atomicAdd(d_stored, 2u);
                                    }
                                }
                         });
            SELF_X_CHECK(cudaDeviceSynchronize());
        }

        unsigned int h_total = 0;
        unsigned int h_stored = 0;
        SELF_X_CHECK(cudaMemcpy(&h_total, d_total, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        SELF_X_CHECK(cudaMemcpy(&h_stored, d_stored, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Persist only what fit; n_intersect = stored count so undo never
        // walks out of bounds.
        if (capacity_ints > 0 && h_stored > 0 && sp->intersected_triangle_idx != nullptr
            && d_intersections != nullptr) {
            unsigned int copy_n = h_stored < capacity_ints ? h_stored : capacity_ints;
            SELF_X_CHECK(cudaMemcpy(sp->intersected_triangle_idx, d_intersections,
                                    (size_t)copy_n * sizeof(unsigned int),
                                    cudaMemcpyDeviceToDevice));
        }
        if (sp->n_intersect != nullptr) {
            SELF_X_CHECK(cudaMemcpy(sp->n_intersect, &h_stored, sizeof(unsigned int),
                                    cudaMemcpyHostToDevice));
        }

        SELF_X_CHECK(cudaMemcpy(&h_isIntersect, d_isIntersect, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        if (h_total > capacity_ints) {
            throw std::runtime_error(
                "self_intersect: intersection result buffer overflow (total=" +
                std::to_string(h_total) + ", capacity=" + std::to_string(capacity_ints) +
                "). Consumers would have read out of bounds; aborting.");
        }

        return h_isIntersect != 0;
    }

}

#endif
