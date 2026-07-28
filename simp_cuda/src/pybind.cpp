#include "cusimp.h"
#include "cusimp_free.h"
#include <torch/extension.h>
#include <cmath>
#include <limits>

namespace cusimp_free
{

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

  static void validate_mesh_tensors(const torch::Tensor &points,
                                    const torch::Tensor &triangles)
  {
    CHECK_INPUT(points);
    CHECK_INPUT(triangles);

    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                "points must have shape [N, 3], got ", points.sizes());
    TORCH_CHECK(triangles.dim() == 2 && triangles.size(1) == 3,
                "triangles must have shape [M, 3], got ", triangles.sizes());
    TORCH_CHECK(points.dtype() == torch::kFloat,
                "points must be float32");
    TORCH_CHECK(triangles.dtype() == torch::kInt,
                "triangles must be int32");
    TORCH_CHECK(points.size(0) > 0, "points must be non-empty");
    TORCH_CHECK(triangles.size(0) > 0, "triangles must be non-empty");
    TORCH_CHECK(points.device() == triangles.device(),
                "points and triangles must be on the same device");

    // Host-side finite / index range checks (copy is small relative to kernels).
    auto points_cpu = points.to(torch::kCPU);
    auto tris_cpu = triangles.to(torch::kCPU);
    auto p_acc = points_cpu.accessor<float, 2>();
    auto t_acc = tris_cpu.accessor<int, 2>();
    const int64_t nPts = points.size(0);
    const int64_t nTris = triangles.size(0);

    for (int64_t i = 0; i < nPts; ++i) {
      for (int c = 0; c < 3; ++c) {
        float v = p_acc[i][c];
        TORCH_CHECK(std::isfinite(v),
                    "points contain non-finite value at row ", i);
      }
    }
    for (int64_t i = 0; i < nTris; ++i) {
      for (int c = 0; c < 3; ++c) {
        int idx = t_acc[i][c];
        TORCH_CHECK(idx >= 0 && idx < nPts,
                    "triangle index out of range at face ", i,
                    " component ", c, ": ", idx, " (n_pts=", nPts, ")");
      }
      TORCH_CHECK(
          t_acc[i][0] != t_acc[i][1] && t_acc[i][1] != t_acc[i][2] &&
              t_acc[i][0] != t_acc[i][2],
          "degenerate triangle with repeated indices at face ", i);
    }
  }

  class CUDSP_Free
  {
    CUSimp_Free pamo;

    void release()
    {
      cudaDeviceSynchronize();
      // cudaFree(nullptr) is a no-op; free every owned buffer once.
      cudaFree(pamo.temp_storage); pamo.temp_storage = nullptr;
      cudaFree(pamo.first_near_tris); pamo.first_near_tris = nullptr;
      cudaFree(pamo.near_tris); pamo.near_tris = nullptr;
      cudaFree(pamo.near_offset); pamo.near_offset = nullptr;
      cudaFree(pamo.first_edge); pamo.first_edge = nullptr;
      cudaFree(pamo.edges); pamo.edges = nullptr;
      cudaFree(pamo.vert_Q); pamo.vert_Q = nullptr;
      cudaFree(pamo.edge_cost); pamo.edge_cost = nullptr;
      cudaFree(pamo.tri_min_cost); pamo.tri_min_cost = nullptr;
      cudaFree(pamo.points); pamo.points = nullptr;
      cudaFree(pamo.pts_occ); pamo.pts_occ = nullptr;
      cudaFree(pamo.pts_map); pamo.pts_map = nullptr;
      cudaFree(pamo.triangles); pamo.triangles = nullptr;
      cudaFree(pamo.n_collapsed); pamo.n_collapsed = nullptr;
      cudaFree(pamo.original_points); pamo.original_points = nullptr;
      cudaFree(pamo.original_tris); pamo.original_tris = nullptr;
      cudaFree(pamo.original_edge_cost); pamo.original_edge_cost = nullptr;
      cudaFree(pamo.collapsed_edge_idx); pamo.collapsed_edge_idx = nullptr;
      cudaFree(pamo.n_edges_undo); pamo.n_edges_undo = nullptr;
      cudaFree(pamo.edges_undo); pamo.edges_undo = nullptr;
      cudaFree(pamo.vertices_undo_list); pamo.vertices_undo_list = nullptr;
      cudaFree(pamo.tmp_vertices_undo_list); pamo.tmp_vertices_undo_list = nullptr;
      cudaFree(pamo.vertices_invalid_list); pamo.vertices_invalid_list = nullptr;
      cudaFree(pamo.vertices_invalid_table); pamo.vertices_invalid_table = nullptr;
      cudaFree(pamo.query_triangle_list); pamo.query_triangle_list = nullptr;
      cudaFree(pamo.intersected_triangle_idx); pamo.intersected_triangle_idx = nullptr;
      cudaFree(pamo.n_intersect); pamo.n_intersect = nullptr;
      // Self-intersect / collapse scratch pool
      cudaFree(pamo.si_is_intersect); pamo.si_is_intersect = nullptr;
      cudaFree(pamo.si_total); pamo.si_total = nullptr;
      cudaFree(pamo.si_stored); pamo.si_stored = nullptr;
      cudaFree(pamo.si_intersections); pamo.si_intersections = nullptr;
      pamo.allocated_si_intersections = 0;
      pamo.allocated_collapsed_edge_idx = 0;
      pamo.allocated_edges_undo = 0;
      pamo.allocated_vertices_undo = 0;
    }

public:
    ~CUDSP_Free()
    {
      release();
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor points, torch::Tensor triangles, torch::Tensor verts_undo,
            int n_verts_undo, float scale, float threshold, bool is_stuck, bool init)
    {
      validate_mesh_tensors(points, triangles);
      CHECK_INPUT(verts_undo);
      TORCH_CHECK(verts_undo.dtype() == torch::kInt,
                  "verts_undo must be int32");
      TORCH_CHECK(n_verts_undo >= 0, "n_verts_undo must be non-negative");
      TORCH_CHECK(n_verts_undo <= verts_undo.numel(),
                  "n_verts_undo exceeds verts_undo length");
      TORCH_CHECK(std::isfinite(scale) && scale > 0.0f,
                  "scale must be finite and positive");
      TORCH_CHECK(std::isfinite(threshold) && threshold >= 0.0f,
                  "threshold must be finite and non-negative");

      torch::ScalarType scalarType = torch::kFloat;
      torch::ScalarType indexType = torch::kInt;

      int nPts = static_cast<int>(points.size(0));
      int nTris = static_cast<int>(triangles.size(0));

      pamo.forward(reinterpret_cast<Vertex<float> *>(points.data_ptr<float>()),
            reinterpret_cast<Triangle<int> *>(triangles.data_ptr<int>()),
            reinterpret_cast<int *>(verts_undo.data_ptr<int>()),
            n_verts_undo,
            nPts, nTris, scale, threshold, is_stuck, init);

      auto opts_f = torch::TensorOptions().device(points.device()).dtype(scalarType);
      auto opts_i = torch::TensorOptions().device(points.device()).dtype(indexType);

      auto verts =
          torch::from_blob(
              pamo.points, torch::IntArrayRef{pamo.n_pts, 3}, opts_f)
              .clone();
      auto tris =
          torch::from_blob(
              pamo.triangles, torch::IntArrayRef{pamo.n_tris, 3}, opts_i)
              .clone();

      auto verts_occ =
          torch::from_blob(
              pamo.pts_occ, torch::IntArrayRef{pamo.n_pts, 1}, opts_i)
              .clone();
      auto verts_map =
          torch::from_blob(
              pamo.pts_map, torch::IntArrayRef{pamo.n_pts, 1}, opts_i)
              .clone();

      auto vertices_undo =
          torch::from_blob(
              pamo.vertices_undo_list, torch::IntArrayRef{pamo.n_vertices_undo}, opts_i)
              .clone();

      return {verts, tris, verts_occ, verts_map, vertices_undo};
    }
  };

} // namespace cusimp_free

namespace cusimp
{

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

  class CUDSP
  {
    CUSimp sp;

    void release()
    {
      cudaDeviceSynchronize();
      cudaFree(sp.temp_storage); sp.temp_storage = nullptr;
      cudaFree(sp.first_near_tris); sp.first_near_tris = nullptr;
      cudaFree(sp.near_tris); sp.near_tris = nullptr;
      cudaFree(sp.near_offset); sp.near_offset = nullptr;
      cudaFree(sp.first_edge); sp.first_edge = nullptr;
      cudaFree(sp.edges); sp.edges = nullptr;
      cudaFree(sp.vert_Q); sp.vert_Q = nullptr;
      cudaFree(sp.edge_cost); sp.edge_cost = nullptr;
      cudaFree(sp.tri_min_cost); sp.tri_min_cost = nullptr;
      cudaFree(sp.points); sp.points = nullptr;
      cudaFree(sp.pts_occ); sp.pts_occ = nullptr;
      cudaFree(sp.pts_map); sp.pts_map = nullptr;
      cudaFree(sp.triangles); sp.triangles = nullptr;
    }

public:
    ~CUDSP()
    {
      release();
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor points, torch::Tensor triangles, float scale, float threshold, bool init)
    {
      CHECK_INPUT(points);
      CHECK_INPUT(triangles);

      TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                  "points must have shape [N, 3]");
      TORCH_CHECK(triangles.dim() == 2 && triangles.size(1) == 3,
                  "triangles must have shape [M, 3]");
      TORCH_CHECK(points.dtype() == torch::kFloat, "points must be float32");
      TORCH_CHECK(triangles.dtype() == torch::kInt, "triangles must be int32");
      TORCH_CHECK(points.size(0) > 0 && triangles.size(0) > 0,
                  "empty mesh not allowed");
      TORCH_CHECK(std::isfinite(scale) && scale > 0.0f,
                  "scale must be finite and positive");

      torch::ScalarType scalarType = torch::kFloat;
      torch::ScalarType indexType = torch::kInt;

      int nPts = static_cast<int>(points.size(0));
      int nTris = static_cast<int>(triangles.size(0));

      sp.forward(reinterpret_cast<Vertex<float> *>(points.data_ptr<float>()),
                  reinterpret_cast<Triangle<int> *>(triangles.data_ptr<int>()),
                  nPts, nTris, scale, threshold, init);

      auto opts_f = torch::TensorOptions().device(points.device()).dtype(scalarType);
      auto opts_i = torch::TensorOptions().device(points.device()).dtype(indexType);

      auto verts =
          torch::from_blob(
              sp.points, torch::IntArrayRef{sp.n_pts, 3}, opts_f)
              .clone();
      auto tris =
          torch::from_blob(
              sp.triangles, torch::IntArrayRef{sp.n_tris, 3}, opts_i)
              .clone();

      auto verts_occ =
          torch::from_blob(
              sp.pts_occ, torch::IntArrayRef{sp.n_pts, 1}, opts_i)
              .clone();
      auto verts_map =
          torch::from_blob(
              sp.pts_map, torch::IntArrayRef{sp.n_pts, 1}, opts_i)
              .clone();

      return {verts, tris, verts_occ, verts_map};
    }
  };

} // namespace cusimp



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  pybind11::class_<cusimp_free::CUDSP_Free>(m, "CUDSP_Free")
      .def(py::init<>())
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, int, float, float, bool, bool>(&cusimp_free::CUDSP_Free::forward));
      
  pybind11::class_<cusimp::CUDSP>(m, "CUDSP")
      .def(py::init<>())
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, float, float, bool>(&cusimp::CUDSP::forward));
}
