#include "g_dbscan.h"

#include <cuda.h>

namespace
{
bool has_nonzero(std::vector<int> &v)
{
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (v[i] > 0)
      return true;
  }
  return false;
}
}

namespace clustering
{

GDBSCAN::GDBSCAN(const Dataset::Ptr dset)
    : m_dset(dset), d_data(0), vA_size(sizeof(int) * dset->rows()), d_Va0(0), d_Va1(0), h_Va0(dset->rows(), 0), h_Va1(dset->rows(), 0), d_Ea(0), d_Fa(0), d_Xa(0), m_fit_time(.0), m_predict_time(.0), core(dset->rows(), false), labels(dset->rows(), -1)
{

  size_t alloc_size = sizeof(float) * m_dset->num_points();
  cudaError_t r = cudaMalloc(reinterpret_cast<void **>(&d_data), alloc_size);

  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda d_data malloc error :" + std::to_string(r));
  }

  LOG(INFO) << "Allocated " << alloc_size << " bytes on device for "
            << m_dset->num_points() << " points";

  r = cudaMalloc(reinterpret_cast<void **>(&d_Va0), vA_size);

  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda d_Va0 malloc error :" + std::to_string(r));
  }

  r = cudaMalloc(reinterpret_cast<void **>(&d_Va1), vA_size);

  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda d_Va1 malloc error :" + std::to_string(r));
  }

  LOG(INFO) << "Allocated " << vA_size << " bytes on device for Va0 and Va1";

  r = cudaMalloc(reinterpret_cast<void **>(&d_Fa), vA_size);

  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda d_Fa malloc error :" + std::to_string(r));
  }

  LOG(INFO) << "Allocated " << vA_size << " bytes on device for d_Fa";

  r = cudaMalloc(reinterpret_cast<void **>(&d_Xa), vA_size);

  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda d_Xa malloc error :" + std::to_string(r));
  }

  LOG(INFO) << "Allocated " << vA_size << " bytes on device for d_Xa";

  const size_t cols = m_dset->cols();
  size_t copysize = cols * sizeof(float);

  for (size_t i = 0; i < m_dset->rows(); ++i)
  {
    r = cudaMemcpy(d_data + i * cols,
                   m_dset->data()[i].data(),
                   copysize,
                   cudaMemcpyHostToDevice);

    if (r != cudaSuccess)
    {
      throw std::runtime_error("Cuda memcpy error :" + std::to_string(r));
    }
    VLOG(3) << "Copied " << i << "th row to device, size = " << copysize;
  }
}

GDBSCAN::~GDBSCAN()
{
  if (d_data)
  {
    cudaFree(d_data);
    d_data = 0;
  }

  if (d_Va0)
  {
    cudaFree(d_Va0);
    d_Va0 = 0;
  }

  if (d_Va1)
  {
    cudaFree(d_Va1);
    d_Va1 = 0;
  }

  if (d_Ea)
  {
    cudaFree(d_Ea);
    d_Ea = 0;
  }

  if (d_Fa)
  {
    cudaFree(d_Fa);
    d_Fa = 0;
  }

  if (d_Xa)
  {
    cudaFree(d_Xa);
    d_Xa = 0;
  }
}

void GDBSCAN::Va_device_to_host()
{
  cudaError_t r = cudaMemcpy(&h_Va0[0], d_Va0, vA_size, cudaMemcpyDeviceToHost);
  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda memcpy Va0 device to host error :" +
                             std::to_string(r));
  }
  r = cudaMemcpy(&h_Va1[0], d_Va1, vA_size, cudaMemcpyDeviceToHost);
  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda memcpy Va1 device to host error :" +
                             std::to_string(r));
  }
}

// dist_type:
//      0 - L2 distance;
//      otherwise - cosine distance, input features must be normalized beforehands.
void GDBSCAN::fit(float eps, size_t min_elems, int dist_type)
{
  const double start = omp_get_wtime();
  // First Step (Vertices degree calculation): For each vertex, we calculate the
  // total number of adjacent vertices. However we can use the multiple cores of
  // the GPU to process multiple vertices in parallel. Our parallel strategy
  // using GPU assigns a thread to each vertex, i.e., each entry of the vector
  // Va. Each GPU thread will count how many adjacent vertex has under its
  // responsibility, filling the first value on the vector Va. As we can see,
  // there are no dependency (or communication) between those parallel tasks
  // (embarrassingly parallel problem). Thus, the computational complexity can
  // be reduced from O(V2) to O(V).

  int N = static_cast<int>(m_dset->rows());
  int colsize = static_cast<int>(m_dset->cols());

  LOG(INFO) << "Starting vertdegree on " << N << "x" << colsize << " "
            << (N + 255) / 256 << "x" << 256;

  vertdegree(N, colsize, eps, d_data, d_Va0, dist_type);

  LOG(INFO) << "Executed vertdegree transfer";

  //   Second Step (Calculation of the adjacency lists indices): The second
  //   value in Va is related to the start
  // index in Ea of the adjacency list of a particular vertex. The calculation
  // of this value depends on the start index of the vertex adjacency list and
  // the degree of the previous vertex. For example, the start index for the
  // vertex 0 is 0, since it is the first vertex. For the vertex 1, the start
  // index is the start index from the previous vertex (i.e. 0), plus its
  // degree, already calculated in the previous step. We realize that we have a
  // data dependency where the next vertex depends on the calculation of the
  // preceding vertices. This is a problem that can be efficiently done in
  // parallel using an exclusive scan operation [23]. For this operation, we
  // used the thrust library, distributed as part of the CUDA SDK. This library
  // provides, among others algorithms, an optimized exclusive scan
  // implementation that is suitable for our method

  adjlistsind(N, d_Va0, d_Va1);

  LOG(INFO) << "Executed adjlistsind transfer";

  Va_device_to_host();

  LOG(INFO) << "Finished transfer";

  for (int i = 0; i < N; ++i)
  {
    if (static_cast<size_t>(h_Va0[i]) >= min_elems)
    {
      core[i] = true;
    }
  }

  //   Third Step (Assembly of adjacency lists): Having the vector Va been
  //   completely filled, i.e., for each
  // vertex, we know its degree and the start index of its adjacency list,
  // calculated in the two previous steps, we can now simply mount the compact
  // adjacency list, represented by Ea. Following the logic of the first step,
  // we assign a GPU thread to each vertex. Each of these threads will fill the
  // adjacency list of its associated vertex with all vertices adjacent to it.
  // The adjacency list for each vertex starts at the indices present in the
  // second value of Va, and has an offset related to the degree of the vertex.

  size_t Ea_size =
      static_cast<size_t>(h_Va0[h_Va0.size() - 1] + h_Va1[h_Va1.size() - 1]) *
      sizeof(int);

  LOG(INFO) << "Allocating " << Ea_size << " bytes for Ea "
            << h_Va0[h_Va0.size() - 1] << "+" << h_Va1[h_Va1.size() - 1];

  if (d_Ea)
  {
    cudaFree(d_Ea);
    d_Ea = 0;
  }

  cudaError_t r = cudaMalloc(reinterpret_cast<void **>(&d_Ea), Ea_size);

  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda d_Ea malloc error :" + std::to_string(r));
  }

  asmadjlist(N, colsize, eps, d_data, d_Va1, d_Ea, dist_type);

  m_fit_time = omp_get_wtime() - start;

  LOG(INFO) << "Executed asmadjlist transfer";
}

void GDBSCAN::Fa_Xa_to_device(const std::vector<int> &Fa, const std::vector<int> &Xa)
{
  cudaError_t r = cudaMemcpy(d_Fa, &Fa[0], vA_size, cudaMemcpyHostToDevice);
  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda memcpy Fa host to device :" +
                             std::to_string(r));
  }
  r = cudaMemcpy(d_Xa, &Xa[0], vA_size, cudaMemcpyHostToDevice);
  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda memcpy Xa host to device :" +
                             std::to_string(r));
  }
}

void GDBSCAN::Xa_to_host(std::vector<int> &Xa)
{
  cudaError_t r = cudaMemcpy(&Xa[0], d_Xa, vA_size, cudaMemcpyDeviceToHost);
  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda memcpy Xa device to host :" +
                             std::to_string(r));
  }
}

void GDBSCAN::Fa_to_host(std::vector<int> &Fa)
{
  cudaError_t r = cudaMemcpy(&Fa[0], d_Fa, vA_size, cudaMemcpyDeviceToHost);
  if (r != cudaSuccess)
  {
    throw std::runtime_error("Cuda memcpy Fa device to host :" +
                             std::to_string(r));
  }
}

void GDBSCAN::breadth_first_search(int i,
                                   int32_t cluster,
                                   std::vector<bool> &visited)
{
  int N = static_cast<int>(m_dset->rows());

  std::vector<int> Xa(m_dset->rows(), 0);
  std::vector<int> Fa(m_dset->rows(), 0);

  Fa[i] = 1;

  Fa_Xa_to_device(Fa, Xa);

  while (has_nonzero(Fa))
  {
    breadth_first_search_kern(N, d_Ea, d_Va0, d_Va1, d_Fa, d_Xa);
    Fa_to_host(Fa);
  }

  Xa_to_host(Xa);

  for (size_t j = 0; j < m_dset->rows(); ++j)
  {
    if (Xa[j])
    {
      visited[j] = true;
      labels[j] = cluster;
      // LOG(INFO) << "Assigning " << j << " " << cluster;
    }
  }
}

int32_t
GDBSCAN::predict()
{
  //   Clusters identification
  // For this step, we decided to parallelize the BFS. Our parallelization
  // approach in CUDA is based on the work presented in [22], which performs a
  // level synchronization, i.e. the BFS traverses the graph in levels. Once a
  // level is visited, it is not visited again. The concept of border in the BFS
  // corresponds to all nodes being processed at the current level. In our
  // implementation we assign one thread to each vertex. Two Boolean vectors,
  // Borders and Visiteds, namely Fa and Xa, respectively, of size V are created
  // to store the vertices that are on the border of BFS (vertices of the
  // current level) and the vertices already visited. In each iteration, each
  // thread (vertex) looks for its entry in the vector Fa. If its position is
  // marked, the vertex removes its own entry on Fa and marks its position in
  // the vector Xa (it is removed from the border, and it has been visited, so
  // we can go to the next level). It also adds its neighbours to the vector Fa
  // if they have not already been visited, thus beginning the search in a new
  // level. This process is repeated until the boundary becomes empty. We
  // illustrate the functioning of our BFS parallel implementation in Algorithm
  // 3 and 4.

  int32_t cluster = 0;
  std::vector<bool> visited(m_dset->rows(), false);

  const double start = omp_get_wtime();

  for (size_t i = 0; i < m_dset->rows(); ++i)
  {
    if (visited[i])
      continue;
    if (!core[i])
      continue;

    visited[i] = true;
    labels[i] = cluster;
    breadth_first_search(static_cast<int>(i), cluster, visited);
    cluster += 1;
  }

  m_predict_time = omp_get_wtime() - start;

  return cluster;
}

const GDBSCAN::Labels &
GDBSCAN::get_labels()
{
  return labels;
}

} // namespace clustering
