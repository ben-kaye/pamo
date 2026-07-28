#include "cusimp_free.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "thrust/fill.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include "bvh/self_intersect.cuh"
#include <fstream>
#include <thrust/shuffle.h>
#include <thrust/random.h>

namespace cusimp_free
{
    constexpr const int BLOCK_SIZE = 512;
    constexpr const float COST_RANGE = 10.0;

    void check_cuda_result(cudaError_t code, const char *file, int line)
    {
        if (code == cudaSuccess)
            return;

        fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file,
                line);
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(code) +
                                 " at " + file + ":" + std::to_string(line));
    }

#define CHECK_CUDA(code) check_cuda_result(code, __FILE__, __LINE__)
#define CHECK_KERNEL() check_cuda_result(cudaGetLastError(), __FILE__, __LINE__)

    template <typename T>
    inline __device__ __host__ T min(T a, T b) { return a < b ? a : b; }
    template <typename T>
    inline __device__ __host__ T max(T a, T b) { return a > b ? a : b; }
    template <typename T>
    inline __device__ __host__ T clamp(T x, T a, T b) { return min(max(a, x), b); }

    void CUSimp_Free::ensure_temp_storage_size(size_t size)
    {
        if (size > allocated_temp_storage_size)
        {
            allocated_temp_storage_size = size_t(size + size / 5);
            // fprintf(stderr, "allocated_temp_storage_size %ld\n", allocated_temp_storage_size);
            CHECK_CUDA(cudaFree(temp_storage));
            CHECK_CUDA(cudaMalloc((void **)&temp_storage, allocated_temp_storage_size));
        }
    }

    void CUSimp_Free::ensure_pts_storage_size(size_t point_count)
    {
        if (point_count > allocated_pts)
        {
            allocated_pts = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_pts %ld\n", allocated_pts);
            CHECK_CUDA(cudaFree(points));
            CHECK_CUDA(
                cudaMalloc((void **)&points, (allocated_pts + 1) * sizeof(Vertex<float>)));

            CHECK_CUDA(cudaFree(pts_occ));
            CHECK_CUDA(
                cudaMalloc((void **)&pts_occ, (allocated_pts + 1) * sizeof(int)));

            CHECK_CUDA(cudaFree(pts_map));
            CHECK_CUDA(
                cudaMalloc((void **)&pts_map, (allocated_pts + 1) * sizeof(int)));
            // restore buffer
            CHECK_CUDA(cudaFree(original_points));
            CHECK_CUDA(
                cudaMalloc((void **)&original_points, (allocated_pts + 1) * sizeof(Vertex<float>)));

            CHECK_CUDA(cudaFree(vertices_undo_list));
            CHECK_CUDA(
                cudaMalloc((void **)&vertices_undo_list, (allocated_pts + 1) * sizeof(int)));
            
            CHECK_CUDA(cudaFree(tmp_vertices_undo_list));
            CHECK_CUDA(
                cudaMalloc((void **)&tmp_vertices_undo_list, (allocated_pts + 1) * sizeof(int)));

            CHECK_CUDA(cudaFree(vertices_invalid_list));
            CHECK_CUDA(
                cudaMalloc((void **)&vertices_invalid_list, (allocated_pts + 1) * sizeof(int)));

            CHECK_CUDA(cudaFree(vertices_invalid_table));
            CHECK_CUDA(
                cudaMalloc((void **)&vertices_invalid_table, (allocated_pts + 1) * sizeof(bool)));
        }
    }

    void CUSimp_Free::ensure_tris_storage_size(size_t tris_count)
    {
        if (tris_count > allocated_tris)
        {
            allocated_tris = size_t(tris_count + tris_count / 5);
            // fprintf(stderr, "allocated_tris %ld\n", allocated_tris);
            CHECK_CUDA(cudaFree(triangles));
            CHECK_CUDA(
                cudaMalloc((void **)&triangles, (allocated_tris + 1) * sizeof(Triangle<int>)));
            // buffer
            CHECK_CUDA(cudaFree(original_tris));
            CHECK_CUDA(
                cudaMalloc((void **)&original_tris, (allocated_tris + 1) * sizeof(Triangle<int>)));

            CHECK_CUDA(cudaFree(intersected_triangle_idx));
            CHECK_CUDA(cudaMalloc((void **)&intersected_triangle_idx, 2 * (allocated_tris + 1) * sizeof(unsigned int)));
            
            // memory for bvh construction and self intersection check
            selfx::ensure_bvh_storage_size(this);
        }
    }

    void CUSimp_Free::ensure_near_count_storage_size(size_t point_count)
    {
        if (point_count > allocated_near_count)
        {
            allocated_near_count = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_near_count %ld\n", allocated_near_count);
            CHECK_CUDA(cudaFree(first_near_tris));
            CHECK_CUDA(
                cudaMalloc((void **)&first_near_tris, (allocated_near_count + 1) * sizeof(int)));
        }
    }

    void CUSimp_Free::ensure_near_tris_storage_size(size_t near_tri_count)
    {
        if (near_tri_count > allocated_near_tris)
        {
            allocated_near_tris = size_t(near_tri_count + near_tri_count / 5);
            // fprintf(stderr, "allocated_near_tris %ld\n", allocated_near_tris);
            CHECK_CUDA(cudaFree(near_tris));
            CHECK_CUDA(
                cudaMalloc((void **)&near_tris, (allocated_near_tris + 1) * sizeof(int)));
        }
    }

    void CUSimp_Free::ensure_near_offset_storage_size(size_t point_count)
    {
        if (point_count > allocated_near_offset)
        {
            allocated_near_offset = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_near_offset %ld\n", allocated_near_offset);
            CHECK_CUDA(cudaFree(near_offset));
            CHECK_CUDA(
                cudaMalloc((void **)&near_offset, (allocated_near_offset + 1) * sizeof(int)));
        }
    }

    void CUSimp_Free::ensure_edge_count_storage_size(size_t tris_count)
    {
        if (tris_count > allocated_edge_count)
        {
            allocated_edge_count = size_t(tris_count + tris_count / 5);
            // fprintf(stderr, "allocated_edge_count %ld\n", allocated_edge_count);
            CHECK_CUDA(cudaFree(first_edge));
            CHECK_CUDA(
                cudaMalloc((void **)&first_edge, (allocated_edge_count + 1) * sizeof(int)));
        }
    }

    void CUSimp_Free::ensure_edge_storage_size(size_t edge_count)
    {
        if (edge_count > allocated_edge)
        {
            allocated_edge = size_t(edge_count + edge_count / 5);
            // fprintf(stderr, "allocated_edge %ld\n", allocated_edge);
            CHECK_CUDA(cudaFree(edges));
            CHECK_CUDA(
                cudaMalloc((void **)&edges, (allocated_edge + 1) * sizeof(Edge<int>)));
        }
    }

    void CUSimp_Free::ensure_vert_Q_storage_size(size_t point_count)
    {
        if (point_count > allocated_vert_Q)
        {
            allocated_vert_Q = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_vert_Q %ld\n", allocated_vert_Q);
            CHECK_CUDA(cudaFree(vert_Q));
            CHECK_CUDA(
                cudaMalloc((void **)&vert_Q, (allocated_vert_Q + 1) * sizeof(Mat4x4<float>)));
        }
    }

    void CUSimp_Free::ensure_edge_cost_storage_size(size_t edge_count)
    {
        if (edge_count > allocated_edge_cost)
        {
            allocated_edge_cost = size_t(edge_count + edge_count / 5);
            // fprintf(stderr, "allocated_edge_cost %ld\n", allocated_edge_cost);
            CHECK_CUDA(cudaFree(edge_cost));
            CHECK_CUDA(
                cudaMalloc((void **)&edge_cost, (allocated_edge_cost + 1) * sizeof(uint32_t)));

            CHECK_CUDA(cudaFree(original_edge_cost));
            CHECK_CUDA(
                cudaMalloc((void **)&original_edge_cost, (allocated_edge_cost + 1) * sizeof(uint32_t)));
        }
    }

    void CUSimp_Free::ensure_tri_min_cost_storage_size(size_t tri_count)
    {
        if (tri_count > allocated_tri_min_cost)
        {
            allocated_tri_min_cost = size_t(tri_count + tri_count / 5);
            // fprintf(stderr, "allocated_tri_min_cost %ld\n", allocated_tri_min_cost);
            CHECK_CUDA(cudaFree(tri_min_cost));
            CHECK_CUDA(
                cudaMalloc((void **)&tri_min_cost, (allocated_tri_min_cost + 1) * sizeof(uint64_cu)));
        }
    }

    // Phase B #4: grow-only collapse/undo scratch (no free+malloc each forward).
    void CUSimp_Free::ensure_collapse_scratch(size_t edge_count)
    {
        if (n_collapsed == nullptr) {
            CHECK_CUDA(cudaMalloc((void **)&n_collapsed, sizeof(int)));
        }
        if (n_intersect == nullptr) {
            CHECK_CUDA(cudaMalloc((void **)&n_intersect, sizeof(unsigned int)));
        }
        if (n_edges_undo == nullptr) {
            CHECK_CUDA(cudaMalloc((void **)&n_edges_undo, sizeof(int)));
        }
        if (edge_count > allocated_collapsed_edge_idx) {
            size_t new_cap = size_t(edge_count + edge_count / 5);
            CHECK_CUDA(cudaFree(collapsed_edge_idx));
            CHECK_CUDA(cudaMalloc((void **)&collapsed_edge_idx, new_cap * sizeof(int)));
            allocated_collapsed_edge_idx = new_cap;
        }
    }

    void CUSimp_Free::ensure_undo_scratch(size_t n_collapsed_count)
    {
        // edges_undo holds 2 ints per collapsed edge.
        size_t need = 2 * n_collapsed_count;
        if (need == 0) {
            return;
        }
        if (need > allocated_edges_undo) {
            size_t new_cap = size_t(need + need / 5);
            CHECK_CUDA(cudaFree(edges_undo));
            CHECK_CUDA(cudaMalloc((void **)&edges_undo, new_cap * sizeof(int)));
            allocated_edges_undo = new_cap;
        }
    }

    __device__
    inline bool is_same_coord(Vertex<float> p, Vertex<float>q){
        return p.x==q.x && p.y==q.y && p.z==q.z;
    }

    __device__
    inline bool detect_line_idx_cu(int i, int j, int k){
        return i==j || j==k || k==i;
    }

    __global__ void remove_invalid_faces(CUSimp_Free sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;
        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        if(detect_line_idx_cu(sp.triangles[tri_index].i, sp.triangles[tri_index].j, sp.triangles[tri_index].k)){
            sp.triangles[tri_index].i = sp.triangles[tri_index].j = sp.triangles[tri_index].k = -1;
            return;
        }
        return;
    }

    __global__ void remove_line_edge_collapse(CUSimp_Free sp){
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;
        // if(index >= *sp.n_collapsed) return;
        Edge<int> edge = sp.edges[edge_index];
        Vertex<float> p = sp.points[edge.u];
        Vertex<float> q = sp.points[edge.v];
        if(!is_same_coord(p,q)) return;

        // collapsing start
        int pos = atomicAdd(sp.n_collapsed, 1);
        sp.collapsed_edge_idx[pos] = edge_index;
        sp.points[edge.v] = {0, 0, 0};
        sp.pts_occ[edge.v] = 0;
        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i == edge.v || sp.triangles[sp.near_tris[i]].j == edge.v || sp.triangles[sp.near_tris[i]].k == edge.v)
            {
                sp.triangles[sp.near_tris[i]].i = sp.triangles[sp.near_tris[i]].j = sp.triangles[sp.near_tris[i]].k = -1;
            }
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i == edge.u || sp.triangles[sp.near_tris[i]].j == edge.u || sp.triangles[sp.near_tris[i]].k == edge.u)
            {
                sp.triangles[sp.near_tris[i]].i = sp.triangles[sp.near_tris[i]].j = sp.triangles[sp.near_tris[i]].k = -1;
            }
            else if (sp.triangles[sp.near_tris[i]].i == edge.v)
                sp.triangles[sp.near_tris[i]].i = edge.u;
            else if (sp.triangles[sp.near_tris[i]].j == edge.v)
                sp.triangles[sp.near_tris[i]].j = edge.u;
            else if (sp.triangles[sp.near_tris[i]].k == edge.v)
                sp.triangles[sp.near_tris[i]].k = edge.u;
        }
    }

    __global__ void count_near_tris_kernel(CUSimp_Free sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        atomicAdd(&sp.first_near_tris[tri.i], 1);
        atomicAdd(&sp.first_near_tris[tri.j], 1);
        atomicAdd(&sp.first_near_tris[tri.k], 1);
    }

    __global__ void create_near_tris_kernel(CUSimp_Free sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        sp.near_tris[sp.first_near_tris[tri.i] + atomicAdd(&sp.near_offset[tri.i], 1)] = tri_index;
        sp.near_tris[sp.first_near_tris[tri.j] + atomicAdd(&sp.near_offset[tri.j], 1)] = tri_index;
        sp.near_tris[sp.first_near_tris[tri.k] + atomicAdd(&sp.near_offset[tri.k], 1)] = tri_index;
    }

    __global__ void count_edge_kernel(CUSimp_Free sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;
        sp.first_edge[tri_index] += (tri.i > tri.j) + (tri.j > tri.k) + (tri.k > tri.i);
    }

    __global__ void create_edge_kernel(CUSimp_Free sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        int first = sp.first_edge[tri_index];
        if (tri.i > tri.j)
            sp.edges[first++] = {int(tri.j), int(tri.i)};
        if (tri.j > tri.k)
            sp.edges[first++] = {int(tri.k), int(tri.j)};
        if (tri.k > tri.i)
            sp.edges[first++] = {int(tri.i), int(tri.k)};
    }

    __device__ Vec4<float> tri2plane(Vertex<float> const *points, Triangle<int> tri)
    {
        Vertex<float> v0 = points[tri.i];
        Vertex<float> v1 = points[tri.j];
        Vertex<float> v2 = points[tri.k];
        Vertex<float> normal = (v1 - v0).cross(v2 - v0);
        normal /= normal.norm();
        float offset = -normal.dot(v0);
        return {normal.x, normal.y, normal.z, offset};
    }

    __global__ void compute_vert_Q_kernel(CUSimp_Free sp)
    {
        int pt_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (pt_index >= sp.n_pts)
            return;
        // printf("pt_index %d sp.n_pts %d\n", pt_index, sp.n_pts);

        int first = sp.first_near_tris[pt_index];
        int last = sp.first_near_tris[pt_index + 1];
        Mat4x4<float> Kp{0};
        for (int i = first; i < last; ++i)
        {
            Vec4<float> p = tri2plane(sp.points, sp.triangles[sp.near_tris[i]]);
            Mat4x4<float> temp = p.dot_T(p);
            Kp += p.dot_T(p);
            // printf("p %f %f %f %f\ntemp %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n\n", p.x, p.y, p.z, p.w, temp.m00, temp.m01, temp.m02, temp.m03, temp.m10, temp.m11, temp.m12, temp.m13, temp.m20, temp.m21, temp.m22, temp.m23, temp.m30, temp.m31, temp.m32, temp.m33);
        }
        sp.vert_Q[pt_index] = Kp;
    }

    __device__ float triangle_area(Vertex<float> p0, Vertex<float> p1, Vertex<float> p2)
    {
        float a = (p0 - p1).norm();
        float b = (p1 - p2).norm();
        float c = (p2 - p0).norm();
        float s = (a + b + c) / 2;
        return sqrt(s * (s - a) * (s - b) * (s - c));
    }

    __device__ float edge_length(Vertex<float> p0, Vertex<float> p1)
    {
        return (p0 - p1).norm();
    }

    __global__ void compute_edge_cost_kernel(CUSimp_Free sp, bool is_stuck)
    {
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;

        // Reject by default; only overwrite after every validity check passes.
        // (Stale/uninitialized costs were a nondeterminism and crash source.)
        const uint32_t COST_INVALID = std::numeric_limits<uint32_t>::max();
        sp.edge_cost[edge_index] = COST_INVALID;

        Edge<int> edge = sp.edges[edge_index];
        Vertex<float> v0 = sp.points[edge.u];
        Vertex<float> v1 = sp.points[edge.v];
        int idx_v0 = edge.u;
        int idx_v1 = edge.v;

        if (sp.vertices_invalid_table[idx_v0] && sp.vertices_invalid_table[idx_v1])
            return;

        // is_stuck only affects which vertices are marked invalid (host path).
        // Always recompute costs: edge indices are rebuilt each forward call, so
        // reusing prior array slots would apply costs to the wrong edges.
        (void)is_stuck;

        // check if the collapse is valid
        int dup_num = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            int idx_u;
            // edge i-j; j-k; k-i
            if (sp.triangles[sp.near_tris[i]].i == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].j;
            else if (sp.triangles[sp.near_tris[i]].j == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].k;
            else if (sp.triangles[sp.near_tris[i]].k == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].i;
            else
                printf("error1\n");
            for (int j = sp.first_near_tris[edge.v]; j < sp.first_near_tris[edge.v + 1]; ++j)
            {
                int idx_v;
                // edge i-j; j-k; k-i
                if (sp.triangles[sp.near_tris[j]].i == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].j;
                else if (sp.triangles[sp.near_tris[j]].j == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].k;
                else if (sp.triangles[sp.near_tris[j]].k == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].i;
                else
                    printf("error2\n");
                if (idx_u == idx_v)
                    dup_num++;
                if (dup_num > 2)
                    return;
            }
        }
        if (dup_num != 2)
            return;

        // compute near edge length
        float edge_len = 0;
        int num_edge = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i > sp.triangles[sp.near_tris[i]].j)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].i] - sp.points[sp.triangles[sp.near_tris[i]].j]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].j > sp.triangles[sp.near_tris[i]].k)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].j] - sp.points[sp.triangles[sp.near_tris[i]].k]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].k > sp.triangles[sp.near_tris[i]].i)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].k] - sp.points[sp.triangles[sp.near_tris[i]].i]).norm();
                num_edge++;
            }
        }
        for (int i = sp.first_near_tris[edge.v]; i < sp.first_near_tris[edge.v + 1]; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i > sp.triangles[sp.near_tris[i]].j)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].i] - sp.points[sp.triangles[sp.near_tris[i]].j]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].j > sp.triangles[sp.near_tris[i]].k)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].j] - sp.points[sp.triangles[sp.near_tris[i]].k]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].k > sp.triangles[sp.near_tris[i]].i)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].k] - sp.points[sp.triangles[sp.near_tris[i]].i]).norm();
                num_edge++;
            }
        }
        if (num_edge <= 0 || !(sp.edge_s > 0.0f))
            return;
        edge_len = edge_len / num_edge / sp.edge_s * sp.tres;
        // printf("edge_len %f\n", edge_len);

        // compute edge length
        edge_len = edge_len + (v0 - v1).norm() / sp.edge_s * sp.tres;

        Vertex<float> v = (v0 + v1) / 2;

        // compute skinny triangle cost Q_a
        // test if the triangle normal is flipped
        float Q_a = 0;
        int num_tri = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            Triangle<int> old_tri = sp.triangles[sp.near_tris[i]];
            // if the triangle is shared by edge.u and edge.v, skip
            if (old_tri.i == edge.v || old_tri.j == edge.v || old_tri.k == edge.v)
                continue;

            Vertex<float> old_v0 = sp.points[old_tri.i];
            Vertex<float> old_v1 = sp.points[old_tri.j];
            Vertex<float> old_v2 = sp.points[old_tri.k];

            Vertex<float> new_v0 = sp.points[old_tri.i];
            Vertex<float> new_v1 = sp.points[old_tri.j];
            Vertex<float> new_v2 = sp.points[old_tri.k];

            // replace edge.u with v
            if (old_tri.i == edge.u)
                new_v0 = v;
            if (old_tri.j == edge.u)
                new_v1 = v;
            if (old_tri.k == edge.u)
                new_v2 = v;

            Vertex<float> old_normal = (old_v1 - old_v0).cross(old_v2 - old_v0);
            old_normal /= old_normal.norm();
            Vertex<float> new_normal = (new_v1 - new_v0).cross(new_v2 - new_v0);
            new_normal /= new_normal.norm();

            // if the normal is flipped, invalid collapse
            if (old_normal.dot(new_normal) < 0)
                return;

            // compute the area of the new triangle
            Q_a += 1.0f - clamp(float(4.0f * sqrt(3.0f) * triangle_area(new_v0, new_v1, new_v2) / pow(edge_length(new_v0, new_v1), 2) + pow(edge_length(new_v1, new_v2), 2) + pow(edge_length(new_v2, new_v0), 2) + 0.0000001f), 0.0f, 1.0f);
            num_tri++;
        }

        for (int i = sp.first_near_tris[edge.v]; i < sp.first_near_tris[edge.v + 1]; ++i)
        {
            Triangle<int> old_tri = sp.triangles[sp.near_tris[i]];
            // if the triangle is shared by edge.u and edge.v, skip
            if (old_tri.i == edge.u || old_tri.j == edge.u || old_tri.k == edge.u)
                continue;

            Vertex<float> old_v0 = sp.points[old_tri.i];
            Vertex<float> old_v1 = sp.points[old_tri.j];
            Vertex<float> old_v2 = sp.points[old_tri.k];

            Vertex<float> new_v0 = sp.points[old_tri.i];
            Vertex<float> new_v1 = sp.points[old_tri.j];
            Vertex<float> new_v2 = sp.points[old_tri.k];

            // replace edge.v with v
            if (old_tri.i == edge.v)
                new_v0 = v;
            if (old_tri.j == edge.v)
                new_v1 = v;
            if (old_tri.k == edge.v)
                new_v2 = v;

            Vertex<float> old_normal = (old_v1 - old_v0).cross(old_v2 - old_v0);
            old_normal /= old_normal.norm();
            Vertex<float> new_normal = (new_v1 - new_v0).cross(new_v2 - new_v0);
            new_normal /= new_normal.norm();

            // if the normal is flipped, invalid collapse
            if (old_normal.dot(new_normal) < 0)
                return;

            // compute the area of the new triangle
            Q_a += 1.0f - clamp(float(4.0f * sqrt(3.0f) * triangle_area(new_v0, new_v1, new_v2) / pow(edge_length(new_v0, new_v1), 2) + pow(edge_length(new_v1, new_v2), 2) + pow(edge_length(new_v2, new_v0), 2) + 0.0000001f), 0.0f, 1.0f);
            num_tri++;
        }
        
        // weight
        const float SKINNY_TRIANGLE_PENALTY = 5.0f;
        Q_a = SKINNY_TRIANGLE_PENALTY * Q_a;

        if (num_tri <= 0)
            return;

        Mat4x4<float> Q = sp.vert_Q[edge.u] + sp.vert_Q[edge.v];
        Vec4<float> v4 = {v.x, v.y, v.z, 1};
        float cost = Q.vTMv(v4) / (sp.edge_s * sp.edge_s);
        float total = cost + edge_len + Q_a / float(num_tri) * sp.tres;
        if (!(total == total)) // NaN
            return;
        sp.edge_cost[edge_index] = uint32_t(clamp(total, 0.0f, COST_RANGE) / COST_RANGE * std::numeric_limits<uint32_t>::max());
    }

    __global__ void propagate_edge_cost_kernel(CUSimp_Free sp)
    {
        uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;

        Edge<int> edge = sp.edges[edge_index];
        uint64_cu cost = (((uint64_cu)sp.edge_cost[edge_index]) << 32) | edge_index;
        // printf("cost %llu edge_index %d, ((uint64_cu)edge_index) << 32 %llu, sp.edge_cost[edge_index] %u \n", cost, edge_index, ((uint64_cu)edge_index) << 32, sp.edge_cost[edge_index]);

        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            atomicMin(&sp.tri_min_cost[sp.near_tris[i]], cost);
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            atomicMin(&sp.tri_min_cost[sp.near_tris[i]], cost);
        }
    }

    __device__
    inline float3 negate_float3(float3 v) {
        return make_float3(-v.x, -v.y, -v.z);
    }

    __device__
    bool is_skinny_triangle(Vertex<float>p1, Vertex<float>q1, Vertex<float>r1){
        float3 pq = make_float3(q1.x - p1.x, q1.y - p1.y, q1.z - p1.z);
        float3 pr = make_float3(r1.x - p1.x, r1.y - p1.y, r1.z - p1.z);
        float3 qr = make_float3(r1.x - q1.x, r1.y - q1.y, r1.z - q1.z);

        float len_pq = sqrtf(dot(pq, pq));
        float len_pr = sqrtf(dot(pr, pr));
        float len_qr = sqrtf(dot(qr, qr));

        float max_length = fmaxf(len_pq, fmaxf(len_pr, len_qr));
        float3 cross_pq_pr = make_float3(
            pq.y * pr.z - pq.z * pr.y,
            pq.z * pr.x - pq.x * pr.z,
            pq.x * pr.y - pq.y * pr.x
        );
        float threshold_area = 1e-6 * max_length * max_length;
        float area = 0.5f * sqrtf(cross_pq_pr.x * cross_pq_pr.x + cross_pq_pr.y * cross_pq_pr.y + cross_pq_pr.z * cross_pq_pr.z);

        return (area < threshold_area);
    }

    __global__ void collapse_edge_kernel(CUSimp_Free sp)
    {
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;
        // printf("edge_index %d\n", edge_index);
        // atomicAdd(&sp.debug[0], 1);

        Edge<int> edge = sp.edges[edge_index];
        uint64_cu cost = (((uint64_cu)sp.edge_cost[edge_index]) << 32) | edge_index;
        // printf("sp.edge_cost[edge_index] %u, sp.collapse_t %u\n", sp.edge_cost[edge_index], sp.collapse_t);
        if (sp.edge_cost[edge_index] > sp.collapse_t)
        {
            // printf("enter\n");
            return;
        }

        Vertex<float> v0 = sp.points[edge.u];
        Vertex<float> v1 = sp.points[edge.v];

        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            // printf("sp.near_tri[i] %d, sp.tri_min_cost[sp.near_tri[i]] %llu, cost %llu\n", sp.near_tris[i], sp.tri_min_cost[sp.near_tris[i]], cost);
            if (sp.tri_min_cost[sp.near_tris[i]] != cost)
            {
                // printf("%d edge %d - %d, sp.tri_min_cost[sp.near_tris[i]] %llu, cost %llu\n", edge_index, edge.u, edge.v, sp.tri_min_cost[sp.near_tris[i]], cost);
                return;
            }
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            // printf("sp.tri_min_cost[i] %llu, cost %llu\n", sp.tri_min_cost[i], cost);
            if (sp.tri_min_cost[sp.near_tris[i]] != cost)
            {
                // printf("%d edge %d - %d, sp.tri_min_cost[sp.near_tris[i]] %llu, cost %llu\n", edge_index, edge.u, edge.v, sp.tri_min_cost[sp.near_tris[i]], cost);
                return;
            }
        }

        // collapsing start
        int pos = atomicAdd(sp.n_collapsed, 1);
        sp.collapsed_edge_idx[pos] = edge_index;

        Vertex<float> v = (v0 + v1) / 2;
        sp.points[edge.u] = v;
        sp.points[edge.v] = {0, 0, 0};
        sp.pts_occ[edge.v] = 0;
        first = sp.first_near_tris[edge.u];
        last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i == edge.v || sp.triangles[sp.near_tris[i]].j == edge.v || sp.triangles[sp.near_tris[i]].k == edge.v)
            {
                sp.triangles[sp.near_tris[i]].i = sp.triangles[sp.near_tris[i]].j = sp.triangles[sp.near_tris[i]].k = -1;
            }
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i == edge.u || sp.triangles[sp.near_tris[i]].j == edge.u || sp.triangles[sp.near_tris[i]].k == edge.u)
            {
                sp.triangles[sp.near_tris[i]].i = sp.triangles[sp.near_tris[i]].j = sp.triangles[sp.near_tris[i]].k = -1;
            }
            else if (sp.triangles[sp.near_tris[i]].i == edge.v)
                sp.triangles[sp.near_tris[i]].i = edge.u;
            else if (sp.triangles[sp.near_tris[i]].j == edge.v)
                sp.triangles[sp.near_tris[i]].j = edge.u;
            else if (sp.triangles[sp.near_tris[i]].k == edge.v)
                sp.triangles[sp.near_tris[i]].k = edge.u;
        }



        // detect skinny and undo
        bool is_skinny = false;
        first = sp.first_near_tris[edge.u];
        last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            if(sp.triangles[sp.near_tris[i]].i == -1) continue;
            if(is_skinny_triangle(sp.points[sp.triangles[sp.near_tris[i]].i], sp.points[sp.triangles[sp.near_tris[i]].j], sp.points[sp.triangles[sp.near_tris[i]].k]))
            {
                is_skinny = true;
                break;
            }
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        if(!is_skinny){
            for (int i = first; i < last; ++i)
            {
                if(sp.triangles[sp.near_tris[i]].i == -1) continue;
                    if(is_skinny_triangle(sp.points[sp.triangles[sp.near_tris[i]].i], sp.points[sp.triangles[sp.near_tris[i]].j], sp.points[sp.triangles[sp.near_tris[i]].k]))
                    {
                        is_skinny = true;
                        break;
                    }
            }
        }
        if(is_skinny){
            sp.pts_occ[edge.v] = 1;
            sp.points[edge.u].x = sp.original_points[edge.u].x;
            sp.points[edge.u].y = sp.original_points[edge.u].y;
            sp.points[edge.u].z = sp.original_points[edge.u].z;
            sp.points[edge.v].x = sp.original_points[edge.v].x;
            sp.points[edge.v].y = sp.original_points[edge.v].y;
            sp.points[edge.v].z = sp.original_points[edge.v].z;


            int first = sp.first_near_tris[edge.u];
            int last = sp.first_near_tris[edge.u + 1];
            for (int i = first; i < last; ++i)
            {
                sp.triangles[sp.near_tris[i]].i = sp.original_tris[sp.near_tris[i]].i;
                sp.triangles[sp.near_tris[i]].j = sp.original_tris[sp.near_tris[i]].j;
                sp.triangles[sp.near_tris[i]].k = sp.original_tris[sp.near_tris[i]].k;
            }

            first = sp.first_near_tris[edge.v];
            last = sp.first_near_tris[edge.v + 1];
            for (int i = first; i < last; ++i)
            {
                sp.triangles[sp.near_tris[i]].i = sp.original_tris[sp.near_tris[i]].i;
                sp.triangles[sp.near_tris[i]].j = sp.original_tris[sp.near_tris[i]].j;
                sp.triangles[sp.near_tris[i]].k = sp.original_tris[sp.near_tris[i]].k;
            }
        }
    }

    __global__ void get_undo_candidate_kernel(CUSimp_Free sp)
    {
        int collapsed_edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (collapsed_edge_index >= *sp.n_collapsed)
            return;

        int collapsed_edge_idx = sp.collapsed_edge_idx[collapsed_edge_index];
        Edge<int> edge = sp.edges[collapsed_edge_idx];
        // n_intersect is a stored count of triangle indices (not pairs).
        // Capacity is 2 * (allocated_tris + 1); never read past it.
        int num_intersect = *sp.n_intersect;
        if (num_intersect < 0)
            num_intersect = 0;
        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        bool flag = false;
        for (int i = first; i < last; ++i)
        {
            int current_near_tris = sp.near_tris[i];
            for (int j = 0; j < num_intersect; ++j)
            {
                if (current_near_tris == sp.intersected_triangle_idx[j])
                {
                    int pos = atomicAdd(sp.n_edges_undo, 1);
                    sp.edges_undo[pos] = collapsed_edge_idx;
                    sp.tmp_vertices_undo_list[2 * pos] = edge.u;
                    sp.tmp_vertices_undo_list[2 * pos + 1] = edge.v;
                    flag = true;
                    break;
                }
            }
            if (flag)
                break;
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            if (flag)
                break;
            int current_near_tris = sp.near_tris[i];
            for (int j = 0; j < num_intersect; ++j)
            {
                if (current_near_tris == sp.intersected_triangle_idx[j])
                {
                    int pos = atomicAdd(sp.n_edges_undo, 1);
                    sp.edges_undo[pos] = collapsed_edge_idx;
                    sp.tmp_vertices_undo_list[2 * pos] = edge.u;
                    sp.tmp_vertices_undo_list[2 * pos + 1] = edge.v;
                    flag = true;
                    break;
                }
            }
        }
    }

    __global__ void undo_collapse_kernel(CUSimp_Free sp)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= *sp.n_edges_undo)
            return;
        // if(index >= *sp.n_collapsed) return;

        int collapsed_edge_idx = sp.edges_undo[index];
        // int collapsed_edge_idx = sp.collapsed_edge_idx[index];
        Edge<int> edge = sp.edges[collapsed_edge_idx];
        // printf("undo %d %d\n", edge.u, edge.v);

        sp.pts_occ[edge.v] = 1;

        sp.points[edge.u].x = sp.original_points[edge.u].x;
        sp.points[edge.u].y = sp.original_points[edge.u].y;
        sp.points[edge.u].z = sp.original_points[edge.u].z;
        sp.points[edge.v].x = sp.original_points[edge.v].x;
        sp.points[edge.v].y = sp.original_points[edge.v].y;
        sp.points[edge.v].z = sp.original_points[edge.v].z;

        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            sp.triangles[sp.near_tris[i]].i = sp.original_tris[sp.near_tris[i]].i;
            sp.triangles[sp.near_tris[i]].j = sp.original_tris[sp.near_tris[i]].j;
            sp.triangles[sp.near_tris[i]].k = sp.original_tris[sp.near_tris[i]].k;
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            sp.triangles[sp.near_tris[i]].i = sp.original_tris[sp.near_tris[i]].i;
            sp.triangles[sp.near_tris[i]].j = sp.original_tris[sp.near_tris[i]].j;
            sp.triangles[sp.near_tris[i]].k = sp.original_tris[sp.near_tris[i]].k;
        }
    }

    __global__ void rearrange_index_of_undo_vertices(CUSimp_Free sp, int first)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= (sp.n_vertices_undo - first)) return;
        //if (index >= 2 * sp.n_edges_undo) return;
        // todo : fix this
        int idx_undo_vertices = sp.tmp_vertices_undo_list[index];
        idx_undo_vertices = sp.pts_map[idx_undo_vertices];
        sp.vertices_undo_list[first + index] = idx_undo_vertices;
    }

    __global__ void compute_invalid_vertices_table(CUSimp_Free sp)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sp.n_invalid_vertices)
            return;

        int idx_invalid = sp.vertices_invalid_list[index];
        //printf("idx_invalid %d\n",idx_invalid);
        sp.vertices_invalid_table[idx_invalid] = true;
    }

    __host__ void CUSimp_Free::forward(Vertex<float> *pts, Triangle<int> *tris, int *verts_undo, int n_verts_undo, int nPts, int nTris, float scale, float threshold, bool is_stuck, bool init)
    {
        float epsilon = 1e-3;
        // processing input vertex map
        n_vertices_undo = 0;
        n_invalid_vertices = 0;
        int first_n_vertices_undo = 0;
        // if is stuck, accumulate the invalid list
        if(is_stuck){
            cudaMemcpy(tmp_vertices_undo_list, verts_undo, n_verts_undo * sizeof(int), cudaMemcpyDeviceToDevice);
            n_vertices_undo += n_verts_undo;
            rearrange_index_of_undo_vertices<<<(n_vertices_undo + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this, first_n_vertices_undo);
            first_n_vertices_undo += n_verts_undo;
        }
        else{
            if(n_verts_undo != 0)
                CHECK_CUDA(cudaMemset(vertices_invalid_table, 0, (allocated_pts + 1) * sizeof(bool)));
        }

        if (n_verts_undo != 0)
        {
            // not stuck, set vertics_invalid_table as all 0
            cudaMemcpy(vertices_invalid_list, verts_undo, n_verts_undo * sizeof(int), cudaMemcpyDeviceToDevice);
            n_invalid_vertices = n_verts_undo;
            compute_invalid_vertices_table<<<(n_invalid_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
        }

        int first_n_edges_undo_init_val = first_n_vertices_undo;

        // start simplifying
        tres = threshold;
        edge_s = scale;
        collapse_t = uint32_t(clamp(threshold, float(0.0), COST_RANGE) / COST_RANGE * std::numeric_limits<uint32_t>::max());
        // printf("collapse_t %u\n", collapse_t);

        resize(nPts, nTris);

        ensure_pts_storage_size(n_pts);
        CHECK_CUDA(cudaMemcpy(points, pts, n_pts * sizeof(Vertex<float>),
                              cudaMemcpyHostToDevice));
        std::vector<int> tmp(n_pts, 1);
        CHECK_CUDA(cudaMemcpy(pts_occ, tmp.data(), n_pts * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(pts_occ + n_pts, 0, sizeof(int)));
        ensure_tris_storage_size(n_tris);
        CHECK_CUDA(cudaMemcpy(triangles, tris, n_tris * sizeof(Triangle<int>),
                              cudaMemcpyHostToDevice));
        if (init){
            thrust::device_ptr<Triangle<int>> thrust_triangles(triangles);
            thrust::default_random_engine rng;
            thrust::shuffle(thrust_triangles, thrust_triangles + n_tris, rng);
        }
        
        // original data to do undo operation
        CHECK_CUDA(cudaMemcpy(original_points, points, n_pts * sizeof(Vertex<float>), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(original_tris, triangles, n_tris * sizeof(Triangle<int>),
                              cudaMemcpyDeviceToDevice));

        size_t temp_storage_bytes = 0;

        // get number of near_tris
        ensure_near_count_storage_size(n_pts);
        CHECK_CUDA(cudaMemset(first_near_tris, 0, (n_pts + 1) * sizeof(int)));
        count_near_tris_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_near_tris, first_near_tris, n_pts + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_near_tris, first_near_tris, n_pts + 1);

        // get near_tris
        CHECK_CUDA(cudaMemcpy(&n_near_tris, first_near_tris + n_pts, sizeof(int),
                              cudaMemcpyDeviceToHost));
        ensure_near_tris_storage_size(n_near_tris);
        ensure_near_offset_storage_size(n_pts);
        CHECK_CUDA(cudaMemset(near_offset, 0, (n_pts + 1) * sizeof(int)));
        create_near_tris_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // get number of edge
        ensure_edge_count_storage_size(n_tris);
        CHECK_CUDA(cudaMemset(first_edge, 0, (n_tris + 1) * sizeof(int)));
        count_edge_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_edge, first_edge, n_tris + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_edge, first_edge, n_tris + 1);

        // get edge
        CHECK_CUDA(cudaMemcpy(&n_edges, first_edge + n_tris, sizeof(int),
                              cudaMemcpyDeviceToHost));
        ensure_edge_storage_size(n_edges);
        create_edge_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // compute cost for each edges
        ensure_vert_Q_storage_size(n_pts);
        compute_vert_Q_kernel<<<(n_pts + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        ensure_edge_cost_storage_size(n_edges);
        // Initialize all costs to invalid before the kernel (defensive against
        // partial writes / future early-exit paths). Kernel also assigns first.
        CHECK_CUDA(cudaMemset(edge_cost, 0xFF, (size_t)n_edges * sizeof(uint32_t)));
        if (n_edges > 0) {
            compute_edge_cost_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this, is_stuck);
            CHECK_KERNEL();
        }
        CHECK_CUDA(cudaMemcpy(original_edge_cost, edge_cost, n_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

        // cost propagate
        ensure_tri_min_cost_storage_size(n_tris);
        std::vector<uint64_cu> temp(n_tris, std::numeric_limits<uint64_cu>::max());
        CHECK_CUDA(cudaMemcpy(tri_min_cost, temp.data(), n_tris * sizeof(uint64_cu), cudaMemcpyHostToDevice));
        propagate_edge_cost_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // collapsed edge — Phase B #4: pool/grow, never free+malloc every iter.
        ensure_collapse_scratch((size_t)std::max(n_edges, 0));
        CHECK_CUDA(cudaMemset(n_collapsed, 0, sizeof(int)));
        if (n_edges > 0 && collapsed_edge_idx != nullptr) {
            CHECK_CUDA(cudaMemset(collapsed_edge_idx, 0, (size_t)n_edges * sizeof(int)));
        }
        if (n_edges > 0) {
            collapse_edge_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
        }

        CHECK_CUDA(cudaMemset(n_intersect, 0, sizeof(unsigned int)));
        CHECK_CUDA(cudaMemset(n_edges_undo, 0, sizeof(int)));

        int h_n_collapsed = 0;
        CHECK_CUDA(cudaMemcpy(&h_n_collapsed, n_collapsed, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_n_collapsed > 0) {
            ensure_undo_scratch((size_t)h_n_collapsed);
            CHECK_CUDA(cudaMemset(edges_undo, 0, 2 * (size_t)h_n_collapsed * sizeof(int)));
        }

        if (h_n_collapsed > 0) {
            remove_invalid_faces<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
            if (n_edges > 0) {
                remove_line_edge_collapse<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
            }
            // self intersection check after collapse (throws on overflow / CUDA error)
            (void)selfx::self_intersect(this, n_pts, n_tris, epsilon);
            CHECK_CUDA(cudaMemset(n_edges_undo, 0, sizeof(int)));
            get_undo_candidate_kernel<<<(h_n_collapsed + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
            CHECK_KERNEL();
            CHECK_CUDA(cudaDeviceSynchronize());

            // get number of edges undo
            int h_n_edges_undo = 0;
            CHECK_CUDA(cudaMemcpy(&h_n_edges_undo, n_edges_undo, sizeof(int), cudaMemcpyDeviceToHost));
            n_vertices_undo += 2 * h_n_edges_undo;
            int i = 0;
            while (h_n_edges_undo != 0) {
                i++;
                undo_collapse_kernel<<<(h_n_collapsed + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
                CHECK_KERNEL();
                rearrange_index_of_undo_vertices<<<(n_vertices_undo + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this, first_n_vertices_undo);
                CHECK_KERNEL();
                CHECK_CUDA(cudaDeviceSynchronize());
                first_n_vertices_undo += 2 * h_n_edges_undo;

                (void)selfx::self_intersect(this, n_pts, n_tris, epsilon);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemset(n_edges_undo, 0, sizeof(int)));
                get_undo_candidate_kernel<<<(h_n_collapsed + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
                CHECK_KERNEL();
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(&h_n_edges_undo, n_edges_undo, sizeof(int), cudaMemcpyDeviceToHost));
                n_vertices_undo += 2 * h_n_edges_undo;

                if (i == 5) break;
            }
        }

        CHECK_CUDA(cudaMemcpy(pts_map, pts_occ, (n_pts + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, pts_map, pts_map, n_pts + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, pts_map, pts_map, n_pts + 1);

        // n_intersect / SI / collapse scratch stay pooled for the next forward().
    }
}