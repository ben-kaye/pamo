#ifndef LBVH_SELF_INTERSECT_CUH
#define LBVH_SELF_INTERSECT_CUH
#include <vector>
#include <iostream>
#include <cmath>
#include "bvh.cuh"
#include "query.cuh"
#include "types.cuh"
#include "predicator.cuh"
#include "tri_tri_3d.cuh"
#include "tri_tri_2d.cuh"

// Max AABB-overlap candidates per face. Each candidate packs as 2 uints
// (query_idx, other_idx), so per-face worst-case write is 2 * BUFFER_SIZE.
#define BUFFER_SIZE 512

using namespace std;

namespace cusimp_free {
    class CUSimp_Free;
}

namespace selfx{
    const int BLOCK_SIZE = 512;

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

    // F is Triangle<int>{i,j,k}; face.i == -1 marks deleted. Query sees this as
    // int* with stride 3 so partner leaves with i==-1 are skipped (do not burn
    // candidate slots on deleted faces clustered at the origin placeholder).
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
            face_i_raw, /*stride=*/3u, (unsigned int)BUFFER_SIZE);
        // Store slot count (2 uints per candidate pair) for exclusive_scan.
        num_found_query_raw[idx] = 2u * num_found;
    }

    __global__ void compute_query_list_kernel(
        cusimp_free::Triangle<int>* F_d_raw,
        lbvh::bvh_device<float, selfx::Triangle<float3>> bvh_dev,
        unsigned int* first_query_result_raw,
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
        lbvh::query_device(bvh_dev, lbvh::overlaps(query_box),
                          intersect_candidates_raw, (unsigned int)idx, first,
                          face_i_raw, /*stride=*/3u, (unsigned int)BUFFER_SIZE);
    }

    void ensure_bvh_storage_size(cusimp_free::CUSimp_Free *sp)
    {
        // resize (not just reserve): self_intersect writes [0, num_faces) and
        // exclusive_scan reads/writes a +1 past-the-end total slot.
        // Candidate packing: 2 uints per pair, up to BUFFER_SIZE pairs/face
        // => worst-case total = allocated_tris * BUFFER_SIZE * 2.
        sp->bvh_triangles.resize(sp->allocated_tris);
        sp->num_found_query.resize(sp->allocated_tris + 1);
        sp->first_query_result.resize(sp->allocated_tris + 1);
        sp->intersect_candidates.resize(
            (size_t)sp->allocated_tris * (size_t)BUFFER_SIZE * 2u);
    }

    __device__
    inline int floor_mean(float v1, float v2, float v3, float v4, float v5, float v6) {
        float mean = (v1 + v2 + v3 + v4 + v5 + v6) / 6.0f;
        return (int)floorf(mean);
    }

    // Function to translate points in all three axes
    __device__
    void translate_coordinates(float3 *p1, float3 *q1, float3 *r1, float3 *p2, float3 *q2, float3 *r2) {
        int mean_x = floor_mean(p1->x, q1->x, r1->x, p2->x, q2->x, r2->x);
        int mean_y = floor_mean(p1->y, q1->y, r1->y, p2->y, q2->y, r2->y);
        int mean_z = floor_mean(p1->z, q1->z, r1->z, p2->z, q2->z, r2->z);

        // Translate all points by subtracting the floored mean values
        p1->x -= mean_x; p1->y -= mean_y; p1->z -= mean_z;
        q1->x -= mean_x; q1->y -= mean_y; q1->z -= mean_z;
        r1->x -= mean_x; r1->y -= mean_y; r1->z -= mean_z;
        p2->x -= mean_x; p2->y -= mean_y; p2->z -= mean_z;
        q2->x -= mean_x; q2->y -= mean_y; q2->z -= mean_z;
        r2->x -= mean_x; r2->y -= mean_y; r2->z -= mean_z;
    }

    bool self_intersect(cusimp_free::CUSimp_Free *sp, unsigned int num_vertices, unsigned int num_faces, float epsilon) {
        if (num_faces == 0)
            return false;

        cusimp_free::Vertex<float>* V_d_raw = sp->points;
        cusimp_free::Triangle<int>* F_d_raw = sp->triangles;
        int* pts_occ_raw = sp->pts_occ;
        int* near_tris_raw = sp->near_tris;
        Triangle<float3>* triangles_d_raw = thrust::raw_pointer_cast(sp->bvh_triangles.data());

        // get triangle data to build bvh -----------------
        // Removed faces are marked with i == -1 (see remove_invalid_faces).
        // Indexing V_d_raw[-1] is a 12-byte OOB before the points allocation
        // (HANDOFF2 §3.1; confirmed by compute-sanitizer memcheck).
        // Placeholders use a tiny non-zero triangle so they don't all share the
        // exact origin AABB; query kernels also skip i==-1 as query and partner.
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
        // get number of intersection candidates (slot counts per face)
        compute_num_of_query_result_kernel<<<n_blocks, BLOCK_SIZE>>>(
            F_d_raw, bvh_dev, num_found_results_raw, num_faces);
        cudaDeviceSynchronize();
        thrust::exclusive_scan(thrust::device, num_found_results_raw,
                               num_found_results_raw + num_faces + 1,
                               sp->first_query_result.data());

        // total packed slots (= 2 * total candidate pairs)
        unsigned int total_slots = 0;
        cudaMemcpy(&total_slots, first_query_result_raw + num_faces,
                   sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Grow candidate buffer if exclusive_scan total exceeds capacity
        // (should not with per-face cap, but keep safe for future cap changes).
        if ((size_t)total_slots > sp->intersect_candidates.size()) {
            sp->intersect_candidates.resize((size_t)total_slots + (size_t)total_slots / 5u + 1u);
        }
        thrust::fill(thrust::device, sp->intersect_candidates.begin(),
                     sp->intersect_candidates.end(), 0xFFFFFFFFu);
        unsigned int* intersect_candidates_raw =
            thrust::raw_pointer_cast(sp->intersect_candidates.data());
        first_query_result_raw = thrust::raw_pointer_cast(sp->first_query_result.data());

        // save data of intersection candidates
        compute_query_list_kernel<<<n_blocks, BLOCK_SIZE>>>(
            F_d_raw, bvh_dev, first_query_result_raw, intersect_candidates_raw, num_faces);
        cudaDeviceSynchronize();

        // Actual tri-tri intersection test based on intersection candidates ----
        unsigned int h_isIntersect = 0;
        unsigned int* d_isIntersect;
        cudaMalloc((void**)&d_isIntersect, sizeof(unsigned int));
        cudaMemcpy(d_isIntersect, &h_isIntersect, sizeof(unsigned int), cudaMemcpyHostToDevice);

        // list of actual intersection pairs
        const int maxIntersections = (int)num_faces;
        unsigned int* d_intersections;
        cudaMalloc((void**)&d_intersections, 2ull * (size_t)maxIntersections * sizeof(unsigned int));
        if (maxIntersections > 0) {
            cudaMemset(d_intersections + 2 * maxIntersections - 2, 0, sizeof(unsigned int));
        }

        unsigned int* d_pos;
        cudaMalloc(&d_pos, sizeof(unsigned int));
        cudaMemset(d_pos, 0, sizeof(unsigned int));

        // actual number of candidate pairs (each pair is 2 slots)
        unsigned int num_query_result = total_slots / 2u;

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<unsigned int>(0),
                         thrust::make_counting_iterator<unsigned int>(num_query_result),
                         [near_tris_raw, pts_occ_raw, epsilon, d_isIntersect, d_pos, d_intersections, maxIntersections, first_query_result_raw, triangles_d_raw, num_found_results_raw, intersect_candidates_raw, F_d_raw] __device__(std::size_t idx) {
                                unsigned int query_idx = intersect_candidates_raw[2 * idx];
                                unsigned int current_idx = intersect_candidates_raw[2 * idx + 1];

                                if(query_idx == 0xFFFFFFFF) return;
                                if(current_idx == 0xFFFFFFFF) return;
                                if(F_d_raw[query_idx].i == -1) return;
                                if(F_d_raw[current_idx].i == -1) return;

                                // Retrieve faces for idx and query_idx
                                cusimp_free::Triangle<int> current_face = F_d_raw[current_idx];
                                cusimp_free::Triangle<int> query_face = F_d_raw[query_idx];

                                Triangle<float3> current_tris = triangles_d_raw[current_idx];
                                Triangle<float3> query_tris = triangles_d_raw[query_idx];


                                int vertices_current[] = {current_face.i, current_face.j, current_face.k};
                                int vertices_query[] = {query_face.i, query_face.j, query_face.k};
                                int num_count = 0;

                                float3 p1,q1,r1,p2,q2,r2;
                                p1 = current_tris.v0;
                                q1 = current_tris.v1;
                                r1 = current_tris.v2;
                                p2 = query_tris.v0;
                                q2 = query_tris.v1;
                                r2 = query_tris.v2;

                                translate_coordinates(&p1, &q1, &r1, &p2, &q2, &r2);

                                // compute number of shared vertex
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

                                

                                // check if coplanar
                                if(is_coplanar(tri_a, tri_b)){
                                    return;
                                    if(num_count == 0){
                                        if(coplanar_without_sharing_test(p1,q1,r1,p2,q2,r2)){
                                            atomicExch(d_isIntersect, 1);
                                            int pos = atomicAdd(d_pos, 2);
                                            if (pos < 2 * maxIntersections - 2) {
                                                d_intersections[pos] = query_idx;
                                                d_intersections[pos + 1] = current_idx;
                                            }
                                        }
                                        return;
                                    }

                                    // vertex sharing
                                    if(num_count == 1){
                                        if(coplanar_vertex_sharing_test(p1,q1,r1,p2,q2,r2,epsilon)){
                                            atomicExch(d_isIntersect, 1);
                                            int pos = atomicAdd(d_pos, 2);
                                            if (pos < 2 * maxIntersections - 2) {
                                                d_intersections[pos] = query_idx;
                                                d_intersections[pos + 1] = current_idx;
                                            }
                                        }
                                        return;
                                    }
                                    // edge sharing
                                    if(num_count == 2){
                                        if(coplanar_same_side_test(p1,q1,r1,p2,q2,r2,epsilon)){
                                            atomicExch(d_isIntersect, 1);
                                            int pos = atomicAdd(d_pos, 2);
                                            if (pos < 2 * maxIntersections - 2) {
                                                d_intersections[pos] = query_idx;
                                                d_intersections[pos + 1] = current_idx;
                                            }
                                        }
                                        return;
                                    }
                                    // identical face
                                    if(num_count == 3){
                                        atomicExch(d_isIntersect, 1);
                                        // check where is intersection -------------------
                                        int pos = atomicAdd(d_pos, 2);
                                        if (pos < 2 * maxIntersections - 2) {
                                            d_intersections[pos] = query_idx;
                                            d_intersections[pos + 1] = current_idx;
                                        }
                                        return;
                                    }
                                    return;
                                }
                                else{
                                    // no coplanar, shared edge
                                    if(num_count == 2){
                                        return; // remove from the test
                                    }
                                    else if(detect_shared_edge_coord(p1,q1,r1,p2,q2,r2)){ // remove from the test
                                        return;
                                    }

                                    float3 source, target;

                                    source = make_float3(1,1,1);
                                    target = make_float3(-1,-1,-1);

                                    float r_i1[3];
                                    float r_i2[3];
                                    // actual intersection test
                                    bool isIntersecting = isect_tri_tri_v3(p1,q1,r1,p2,q2,r2,r_i1,r_i2);
                                    
                                    if(isIntersecting){
                                        copy_v3_v3_float3_float(source, r_i1);
                                        copy_v3_v3_float3_float(target, r_i2);
                                        float dist = largest_distance(source, target);
                                        bool sharedVertex = (num_count == 1);
                                        // if the distance is less than eps with shared vertex, the intersection point would be shared vertex
                                        if(dist < epsilon && sharedVertex){
                                            return; // not self intersect
                                        }
                                        else{
                                            atomicExch(d_isIntersect, 1);
                                            int pos = atomicAdd(d_pos, 2);
                                            if (pos < 2 * maxIntersections - 2) {
                                                d_intersections[pos] = query_idx;
                                                d_intersections[pos + 1] = current_idx;
                                            }
                                        }
                                    }
                                    return;             

                                }

                            return;
                         });

        // copy result
        cudaMemcpy(sp->intersected_triangle_idx, d_intersections, 2 * maxIntersections * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(sp->n_intersect, d_pos, sizeof(unsigned int), cudaMemcpyDeviceToDevice);

        cudaFree(d_intersections);
        cudaFree(d_pos);

        cudaDeviceSynchronize();
        cudaMemcpy(&h_isIntersect, d_isIntersect, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaFree(d_isIntersect);

        return h_isIntersect;
    }

}

#endif