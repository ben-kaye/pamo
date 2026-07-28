#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>
#include "bvh/bvh.cuh"
#include "bvh/aabb.cuh"

namespace selfx{
  template <typename T>
  struct Triangle
  {
      T v0, v1, v2;

      inline __device__ __host__ T *data_ptr() { return &v0; }
  };
}

namespace cusimp_free
{
  typedef unsigned long long int uint64_cu;
  // typedef uint64_t uint64_cu;

  template <typename T>
  struct Vertex
  {
    T x, y, z;

    inline __device__ __host__ T *data_ptr() { return &x; }

    inline __device__ __host__ Vertex<T> operator+(Vertex<T> const &other) const
    {
      return {x + other.x, y + other.y, z + other.z};
    }
    inline __device__ __host__ T dot(Vertex<T> const &other) const
    {
      return x * other.x + y * other.y + z * other.z;
    }
    inline __device__ __host__ Vertex<T> cross(Vertex<T> const &other) const
    {
      return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }
    inline __device__ __host__ T norm() const
    {
      return sqrt(x * x + y * y + z * z);
    }
    inline __device__ __host__ Vertex<T> operator-(Vertex<T> const &other) const
    {
      return {x - other.x, y - other.y, z - other.z};
    }

    inline __device__ __host__ Vertex<T> operator*(Vertex<T> const &other) const
    {
      return {x * other.x, y * other.y, z * other.z};
    }

    inline __device__ __host__ Vertex<T> operator*(T const &scalar) const
    {
      return {x * scalar, y * scalar, z * scalar};
    }

    inline __device__ __host__ Vertex<T> operator/(T const &scalar) const
    {
      return {x / scalar, y / scalar, z / scalar};
    }

    inline __device__ __host__ Vertex<T> &operator+=(Vertex<T> const &other)
    {
      x += other.x;
      y += other.y;
      z += other.z;
      return *this;
    }

    inline __device__ __host__ Vertex<T> &operator-=(Vertex<T> const &other)
    {
      x -= other.x;
      y -= other.y;
      z -= other.z;
      return *this;
    }

    inline __device__ __host__ Vertex<T> &operator*=(T const &scalar)
    {
      x *= scalar;
      y *= scalar;
      z *= scalar;
      return *this;
    }

    inline __device__ __host__ Vertex<T> &operator/=(T const &scalar)
    {
      x /= scalar;
      y /= scalar;
      z /= scalar;
      return *this;
    }
  };

  template <typename T>
  struct Edge
  {
    T u, v;
    inline __device__ __host__ T *data_ptr() { return &u; }
  };

  template <typename T>
  struct Triangle
  {
    T i, j, k;
    inline __device__ __host__ T *data_ptr() { return &i; }
  };

  template <typename T>
  struct Mat4x4;

  template <typename T>
  struct Vec4
  {
    T x, y, z, w;
    inline __device__ __host__ T *data_ptr() { return &x; }

    inline __device__ __host__ Mat4x4<T> dot_T(Vec4<T> const &other) const
    {
      return {x * other.x, x * other.y, x * other.z, x * other.w,
              y * other.x, y * other.y, y * other.z, y * other.w,
              z * other.x, z * other.y, z * other.z, z * other.w,
              w * other.x, w * other.y, w * other.z, w * other.w};
    }

    inline __device__ __host__ T dot(Vec4<T> const &other) const
    {
      return x * other.x + y * other.y + z * other.z + w * other.w;
    }
  };

  template <typename T>
  struct Mat4x4
  {
    T m00, m01, m02, m03;
    T m10, m11, m12, m13;
    T m20, m21, m22, m23;
    T m30, m31, m32, m33;
    inline __device__ __host__ T *data_ptr() { return &m00; }
    inline __device__ __host__ T vTMv(Vec4<T> const &other) const
    {
      Vec4<T> vec1x4 = {m00 * other.x + m10 * other.y + m20 * other.z + m30 * other.w,
                     m01 * other.x + m11 * other.y + m21 * other.z + m31 * other.w,
                     m02 * other.x + m12 * other.y + m22 * other.z + m32 * other.w,
                     m03 * other.x + m13 * other.y + m23 * other.z + m33 * other.w};
      return vec1x4.dot(other);
    }

    inline __device__ __host__ Mat4x4<T> operator+(Mat4x4<T> const &other) const
    {
      return {m00 + other.m00, m01 + other.m01, m02 + other.m02, m03 + other.m03,
              m10 + other.m10, m11 + other.m11, m12 + other.m12, m13 + other.m13,
              m20 + other.m20, m21 + other.m21, m22 + other.m22, m23 + other.m23,
              m30 + other.m30, m31 + other.m31, m32 + other.m32, m33 + other.m33};
    }

    inline __device__ __host__ Mat4x4<T> &operator+=(Mat4x4<T> const &other)
    {
      m00 += other.m00;
      m01 += other.m01;
      m02 += other.m02;
      m03 += other.m03;
      m10 += other.m10;
      m11 += other.m11;
      m12 += other.m12;
      m13 += other.m13;
      m20 += other.m20;
      m21 += other.m21;
      m22 += other.m22;
      m23 += other.m23;
      m30 += other.m30;
      m31 += other.m31;
      m32 += other.m32;
      m33 += other.m33;
      return *this;
    }
  };

  struct CUSimp_Free
  {
    float tres{};
    uint32_t collapse_t{};
    float edge_s{};
    int n_pts{};
    int n_tris{};
    int n_edges{};
    int n_near_tris{};

    int n_invalid_vertices{}; // number of invalid vertices from iteration before
    int n_vertices_undo{};

    int* debug{};

    // temp storage
    size_t allocated_temp_storage_size{};
    int *__restrict__ temp_storage{}; // used for prefix sum

    // Vertex -> incident faces (compressed sparse row): faces of v are
    //   near_tris[first_near_tris[v] .. first_near_tris[v+1])
    // near_offset is a per-vertex fill cursor used only while building that layout.
    size_t allocated_near_count{};
    int *__restrict__ first_near_tris{};
    size_t allocated_near_tris{};
    int *__restrict__ near_tris{};
    size_t allocated_near_offset{};
    int *__restrict__ near_offset{};

    // Per-face edge prefixes (first_edge) then flat undirected edge list.
    size_t allocated_edge_count{};
    int *__restrict__ first_edge{};
    size_t allocated_edge{};
    Edge<int> *__restrict__ edges{};

    // Quadric error metric: per-vertex Q, per-edge cost (32-bit unsigned),
    // per-face min packed cost (high 32 = cost, low 32 = edge index) for
    // independent-set selection.
    size_t allocated_vert_Q{};
    Mat4x4<float> *__restrict__ vert_Q{};
    size_t allocated_edge_cost{};
    uint32_t *__restrict__ edge_cost{};
    size_t allocated_tri_min_cost{};
    uint64_cu *__restrict__ tri_min_cost{};

    // Working mesh (starts as input; collapse rewrites in place).
    size_t allocated_pts{};
    Vertex<float> *__restrict__ points{};
    int *__restrict__ pts_occ{}; // 1 = live vertex, 0 = collapsed away
    int *__restrict__ pts_map{}; // exclusive_sum(pts_occ) compact indices
    size_t allocated_tris{};
    Triangle<int> *__restrict__ triangles{}; // deleted faces: i=j=k=-1

    // Collapse + undo. original_* is the snapshot taken at the start of forward().
    int *n_collapsed{};
    Vertex<float> *__restrict__ original_points{};
    Triangle<int> *__restrict__ original_tris{};
    uint32_t *__restrict__ original_edge_cost{};

    int *__restrict__ collapsed_edge_idx{}; // edges accepted by collapse_edge_kernel
    size_t allocated_collapsed_edge_idx{};
    int * n_edges_undo{}; // how many need self-intersection undo
    int * edges_undo{};   // edge indices to restore
    size_t allocated_edges_undo{};

    int *__restrict__ vertices_undo_list{};
    int *__restrict__ tmp_vertices_undo_list{};
    size_t allocated_vertices_undo{}; // capacity for undo lists (can exceed n_pts)
    int *__restrict__ vertices_invalid_list{}; // endpoints blocked on next host iter
    bool *__restrict__ vertices_invalid_table{};

    // Self-intersection output: flat face-index list (length n_intersect)
    // that failed the triangle-triangle test.
    int * __restrict__ query_triangle_list{};
    unsigned int *__restrict__ intersected_triangle_idx{};
    int * n_intersect{};

    // Pooled self-intersect scratch (reused across calls; grown as needed).
    unsigned int *si_is_intersect{};
    unsigned int *si_total{};
    unsigned int *si_stored{};
    unsigned int *si_intersections{};
    size_t allocated_si_intersections{};

    // LBVH broad-phase buffers (two-pass count/scan/fill of AABB candidates).
    thrust::device_vector<selfx::Triangle<float3>> bvh_triangles;
    thrust::device_vector<unsigned int> num_found_query;   // per-face slot counts
    thrust::device_vector<unsigned int> first_query_result; // exclusive_scan offsets
    thrust::device_vector<unsigned int> intersect_candidates; // packed (query, partner) pairs


    inline __host__ void resize(int nPts, int nTris)
    {
      n_pts = nPts;
      n_tris = nTris;
    }

    __host__ void ensure_temp_storage_size(size_t size);
    __host__ void ensure_pts_storage_size(size_t n_pts);
    __host__ void ensure_tris_storage_size(size_t n_tris);
    __host__ void ensure_near_count_storage_size(size_t n_pts);
    __host__ void ensure_near_tris_storage_size(size_t n_near_tris);
    __host__ void ensure_near_offset_storage_size(size_t n_pts);
    __host__ void ensure_edge_count_storage_size(size_t n_tris);
    __host__ void ensure_edge_storage_size(size_t n_edges);
    __host__ void ensure_vert_Q_storage_size(size_t n_pts);
    __host__ void ensure_edge_cost_storage_size(size_t n_edges);
    __host__ void ensure_tri_min_cost_storage_size(size_t n_tris);
    __host__ void ensure_collapse_scratch(size_t n_edges);
    __host__ void ensure_undo_scratch(size_t n_collapsed);
    __host__ void ensure_vertices_undo_storage(size_t n);

    // triangles must start from 0
    __host__ void forward(Vertex<float> *pts, Triangle<int> *tris, int* verts_undo, int n_verts_undo, int nPts, int nTris, float scale, float threshold, bool is_stuck, bool init);
  };
}

