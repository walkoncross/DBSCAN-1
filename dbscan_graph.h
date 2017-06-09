#ifndef DBSCAN_GRAPH_H
#define DBSCAN_GRAPH_H

#include "dataset.h"

namespace {
bool
has_nonzero( std::vector< uint8_t >& v )
{
    for ( size_t i = 0; i < v.size(); ++i ) {
        if ( v[i] > 0 )
            return true;
    }
    return false;
}
}

namespace clustering {
class DBSCAN_GRAPH : private boost::noncopyable {
public:
    typedef std::vector< int32_t > Labels;
    typedef boost::shared_ptr< DBSCAN_GRAPH > Ptr;

private:
    // const Dataset::Ptr m_dset;

    const Dataset::Ptr m_dset;
    std::vector< uint32_t > Va0;
    std::vector< uint64_t > Va1;
    std::vector< uint32_t > Ea;
    std::vector< uint8_t > Fa;
    std::vector< uint8_t > Xa;

    double m_fit_time;
    double m_predict_time;

    std::vector< bool > core;

    Labels labels;

    void Va_device_to_host();
    void Fa_to_host( std::vector< int >& Fa );
    void Xa_to_host( std::vector< int >& Xa );
    void Fa_Xa_to_device( const std::vector< int >& Fa, const std::vector< int >& Xa );
    void breadth_first_search( int i, int32_t cluster, std::vector< bool >& visited );

    static inline float dist( const Eigen::VectorXf& p1, const Eigen::VectorXf& p2 )
    {
        return ( p1 - p2 ).norm();
    }

    template < typename T >
    void prefixsum_inplace( std::vector< T >& x )
    {
        std::vector< T > suma;
        size_t N = x.size();
#pragma omp parallel
        {
            const size_t ithread = omp_get_thread_num();
            const size_t nthreads = omp_get_num_threads();
#pragma omp single
            {
                suma.resize( nthreads + 1 );
                suma[0] = 0;
            }
            T sum = 0;
#pragma omp for schedule( static )
            for ( size_t i = 0; i < N; ++i ) {
                sum += x[i];
                x[i] = sum;
            }
            suma[ithread + 1] = sum;
#pragma omp barrier
            T offset = 0;
            for ( size_t i = 0; i < ( ithread + 1 ); ++i ) {
                offset += suma[i];
            }
#pragma omp for schedule( static )
            for ( size_t i = 0; i < N; ++i ) {
                x[i] += offset;
            }
        }
    }

    void sub_Va()
    {
#pragma omp parallel for
        for ( size_t i = 0; i < Va0.size(); ++i ) {
            Va1[i] -= Va0[i];
        }
    }

    void vertdegree( float eps, size_t min_elems )
    {
        const Dataset::DataContainer& d = m_dset->data();
#pragma omp parallel for
        for ( size_t i = 0; i < m_dset->rows(); ++i ) {
            Va0[i] = 0;

            for ( size_t j = 0; j < m_dset->rows(); ++j ) {
                const float dst = dist( d[i], d[j] );
                if ( dst < eps ) {
                    Va0[i] += 1;
                }
            }
        }
    }

    void asmadjlist( float eps, size_t min_elems )
    {
        const Dataset::DataContainer& d = m_dset->data();
#pragma omp parallel for
        for ( size_t i = 0; i < m_dset->rows(); ++i ) {
            uint64_t basei = Va1[i];
            for ( size_t j = 0; j < m_dset->rows(); ++j ) {
                const float dst = dist( d[i], d[j] );

                if ( dst < eps ) {
                    Ea[basei] = j;
                    ++basei;
                }
            }
        }
    }

    void breadth_first_search_kern()
    {
#pragma omp parallel for
        for ( size_t tid = 0; tid < m_dset->rows(); ++tid ) {
            if ( Fa[tid] ) {
                Fa[tid] = 0;
                Xa[tid] = 1;

                uint32_t nmax_idx = Va1[tid] + Va0[tid];

                for ( uint32_t i = Va1[tid]; i < nmax_idx; ++i ) {
                    uint32_t nid = Ea[i];
                    if ( !Xa[nid] ) {
                        Fa[nid] = 1;
                    }
                }
            }
        }
    }

    void breadth_first_search( size_t i,
                               int32_t cluster,
                               std::vector< bool >& visited )
    {
        std::fill( Fa.begin(), Fa.end(), 0 );
        std::fill( Xa.begin(), Xa.end(), 0 );

        Fa[i] = 1;

        while ( has_nonzero( Fa ) ) {
            breadth_first_search_kern();
        }

        for ( size_t j = 0; j < m_dset->rows(); ++j ) {
            if ( Xa[j] ) {
                visited[j] = true;
                labels[j] = cluster;
                // LOG(INFO) << "Assigning " << j << " " << cluster;
            }
        }
    }

public:
    DBSCAN_GRAPH( const Dataset::Ptr dset )
        : m_dset( dset )
        , Va0( dset->rows(), 0 )
        , Va1( dset->rows(), 0 )
        , Fa( dset->rows(), 0 )
        , Xa( dset->rows(), 0 )
        , m_fit_time( .0 )
        , m_predict_time( .0 )
        , core( dset->rows(), false )
        , labels( dset->rows(), -1 )
    {
    }
    ~DBSCAN_GRAPH()
    {
    }

    const std::vector< uint32_t >& get_Va0() const
    {
        return Va0;
    }
    const std::vector< uint64_t >& get_Va1() const
    {
        return Va1;
    }

    void fit( float eps, size_t min_elems )
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

        LOG( INFO ) << "Graph clustering start";

        vertdegree( eps, min_elems );

        LOG( INFO ) << "Executed vertdegree transfer";

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

        std::copy( Va0.begin(), Va0.end(), Va1.begin() );
        prefixsum_inplace( Va1 );
        sub_Va();

        LOG( INFO ) << "Executed adjlistsind transfer";

        for ( size_t i = 0; i < m_dset->rows(); ++i ) {
            VLOG( 2 ) << Va0[i] << " " << Va1[i];
            core[i] = false;
            if ( static_cast< size_t >( Va0[i] ) >= min_elems ) {
                core[i] = true;
            }
        }

        // //   Third Step (Assembly of adjacency lists): Having the vector Va been
        // //   completely filled, i.e., for each
        // // vertex, we know its degree and the start index of its adjacency list,
        // // calculated in the two previous steps, we can now simply mount the compact
        // // adjacency list, represented by Ea. Following the logic of the first step,
        // // we assign a GPU thread to each vertex. Each of these threads will fill the
        // // adjacency list of its associated vertex with all vertices adjacent to it.
        // // The adjacency list for each vertex starts at the indices present in the
        // // second value of Va, and has an offset related to the degree of the vertex.

        size_t Ea_size = static_cast< size_t >( Va0[Va0.size() - 1] + Va1[Va1.size() - 1] );
        size_t Ea_size_bytes = Ea_size * sizeof( uint32_t );

        LOG( INFO ) << "Allocating for Ea " << ( Ea_size_bytes / 1024.0 / 1024.0 ) << " Mb " << Ea_size << " elements";
        Ea.resize( Ea_size );

        asmadjlist( eps, min_elems );

        m_fit_time = omp_get_wtime() - start;

        LOG( INFO ) << "Executed asmadjlist transfer";
    }

    int32_t predict()
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
        std::vector< bool > visited( m_dset->rows(), false );

        const double start = omp_get_wtime();

        for ( size_t i = 0; i < m_dset->rows(); ++i ) {
            if ( visited[i] )
                continue;
            if ( !core[i] )
                continue;

            visited[i] = true;
            labels[i] = cluster;
            breadth_first_search( i, cluster, visited );
            cluster += 1;
        }

        m_predict_time = omp_get_wtime() - start;

        return cluster;
    }
    const Labels& get_labels()
    {
        return labels;
    }

    const double get_fit_time() const
    {
        return m_fit_time;
    }

    const double get_predict_time() const
    {
        return m_predict_time;
    }
};
}

#endif // DBSCAN_GRAPH_H
