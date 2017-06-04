// Guilherme Andrade, Gabriel Ramos, Daniel Madeira, Rafael Sachetto, Renato
// Ferreira, Leonardo Rocha, G-DBSCAN: A GPU Accelerated Algorithm for
// Density-based Clustering, Procedia Computer Science, Volume 18, 2013, Pages
// 369-378, ISSN 1877-0509, http://dx.doi.org/10.1016/j.procs.2013.05.200.
// (http://www.sciencedirect.com/science/article/pii/S1877050913003438)
// Abstract: With the advent of Web 2.0, we see a new and differentiated
// scenario: there is more data than that can be effectively analyzed.
// Organizing this data has become one of the biggest problems in Computer
// Science. Many algorithms have been proposed for this purpose, highlighting
// those related to the Data Mining area, specifically the clustering
// algorithms. However, these algo- rithms are still a computational challenge
// because of the volume of data that needs to be processed. We found in the
// literature some proposals to make these algorithms feasible, and, recently,
// those related to parallelization on graphics processing units (GPUs) have
// presented good results. In this work we present the G-DBSCAN, a GPU parallel
// version of one of the most widely used clustering algorithms, the DBSCAN.
// Although there are other parallel versions of this algorithm, our technique
// distinguishes itself by the simplicity with which the data are indexed, using
// graphs, allowing various parallelization opportu- nities to be explored. In
// our evaluation we show that the G-DBSCAN using GPU, can be over 100x faster
// than its sequential version using CPU. Keywords: Clustering; Dbscan; Parallel
// computing; GPU

#ifndef G_DBSCAN
#define G_DBSCAN

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dataset.h"

namespace clustering {

void
vertdegree(int N, int colsize, float eps, float* d_data, int* d_Va);

void
adjlistsind(int N, int* Va0, int* Va1);

void
asmadjlist(int N, int colsize, float eps, float* d_data, int* d_Va1, int* d_Ea);

void
breadth_first_search_kern(int N,
                          int* d_Ea,
                          int* d_Va0,
                          int* d_Va1,
                          int* d_Fa,
                          int* d_Xa);

class GDBSCAN : private boost::noncopyable
{
public:
  typedef std::vector<int32_t> Labels;
  typedef boost::shared_ptr<GDBSCAN> Ptr;

private:
  // const Dataset::Ptr m_dset;

  const Dataset::Ptr m_dset;
  float* d_data;
  const size_t vA_size;
  int* d_Va0;
  int* d_Va1;
  std::vector<int> h_Va0;
  std::vector<int> h_Va1;
  int* d_Ea;
  int* d_Fa;
  int* d_Xa;
  double m_fit_time;
  double m_predict_time;
  std::vector<bool> core;
  Labels labels;

  void Va_device_to_host();
  void Fa_to_host(std::vector<int>& Fa);
  void Xa_to_host(std::vector<int>& Xa);
  void Fa_Xa_to_device(const std::vector<int>& Fa, const std::vector<int>& Xa);
  void breadth_first_search(int i, int32_t cluster, std::vector<bool>& visited);

public:
  GDBSCAN(const Dataset::Ptr dset);
  ~GDBSCAN();

  void fit(float eps, size_t min_elems);
  int32_t predict();
  const Labels& get_labels();

  const double get_fit_time() const { return m_fit_time; }

  const double get_predict_time() const { return m_predict_time; }
};
} // namespace clustering

#endif