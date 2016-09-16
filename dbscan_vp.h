#ifndef DBSCAN_VP_H
#define DBSCAN_VP_H

#include "vptree.h"
#include <Eigen/Dense>

namespace clustering {
class DBSCAN_VP {
private:
    static inline double dist( const Eigen::VectorXf& p1, const Eigen::VectorXf& p2 )
    {
        return ( p1 - p2 ).norm();
    }

    typedef VPTREE< Eigen::VectorXf, dist > TVpTree;

public:
    typedef std::vector< int32_t > Labels;
    typedef boost::shared_ptr< DBSCAN_VP > Ptr;

    DBSCAN_VP( double eps, size_t min_elems, int num_threads = 0 )
        : m_eps( eps )
        , m_min_elems( min_elems )
        , m_num_threads( num_threads )
    {
    }
    DBSCAN_VP()
        : m_eps( .0 )
        , m_min_elems( 0 )
        , m_num_threads( 0 )
    {
    }
    ~DBSCAN_VP()
    {
    }

    void init( double eps, size_t min_elems, int num_threads = 0 )
    {
        m_eps = eps;
        m_min_elems = min_elems;
        m_num_threads = 0;
    }

    void find_neighbors( const Dataset::DataContainer& d, uint32_t pid, TVpTree::TNeighborsList& neighbors )
    {
        neighbors.clear();
        m_vp_tree->search( d[pid], m_eps, neighbors );

        // std::cout << "Searching neighbors " << pid << " " << neighbors.size() << std::endl;
        // for ( size_t i = 0; i < neighbors.size(); ++i ) {
        //     std::cout << "\t N" << neighbors[i] << " " << distances[i] << std::endl;
        // }
    }

    void fit( const Dataset::Ptr dset )
    {
        const Dataset::DataContainer& d = dset->data();

        m_vp_tree = boost::make_shared< TVpTree >();
        m_vp_tree->create( dset );

        const size_t dlen = d.size();

        prepare_labels( dlen );

        std::vector< uint32_t > candidates;
        std::vector< uint32_t > new_candidates;

        uint32_t cluster_id = 0;

        TVpTree::TNeighborsList index_neigh;
        TVpTree::TNeighborsList n_neigh;
        TVpTree::TNeighborsList c_neigh;

        for ( uint32_t pid = 0; pid < dlen; ++pid ) {
            if ( m_labels[pid] >= 0 )
                continue;

            find_neighbors( d, pid, index_neigh );

            // std::cout << "Analyzing pid " << pid << " Neigh size " << index_neigh.size() << std::endl;

            if ( index_neigh.size() < m_min_elems )
                continue;

            m_labels[pid] = cluster_id;

            candidates.clear();
            candidates.push_back( pid );

            while ( candidates.size() > 0 ) {
                // std::cout << "\tcandidates = " << candidates.size() << std::endl;

                new_candidates.clear();

                for ( const auto& c_pid : candidates ) {

                    find_neighbors( d, c_pid, c_neigh );

                    // std::cout << "\tAnalyzing c_pid " << c_pid << " c_Neigh size " << c_neigh.size() << std::endl;

                    for ( const auto& nn : c_neigh ) {

                        if ( m_labels[nn.first] >= 0 )
                            continue;

                        m_labels[nn.first] = cluster_id;

                        find_neighbors( d, nn.first, n_neigh );

                        // std::cout << "\t\tAnalyzing nn_pid " << nn.first << " nn_Neigh size " << n_neigh.size() << std::endl;

                        if ( n_neigh.size() >= m_min_elems ) {
                            new_candidates.push_back( nn.first );
                        }
                    }
                }

                // std::cout << "\tnew candidates = " << new_candidates.size() << std::endl;

                candidates = new_candidates;
            }
            ++cluster_id;
        }
    }

    void reset()
    {
        m_labels.clear();
    }

    const Labels& get_labels() const
    {
        return m_labels;
    }

private:
    double m_eps;
    size_t m_min_elems;
    int m_num_threads;

    Labels m_labels;

    void prepare_labels( size_t s )
    {
        m_labels.resize( s );

        for ( auto& l : m_labels ) {
            l = -1;
        }
    }

    TVpTree::Ptr m_vp_tree;
};

// std::ostream& operator<<( std::ostream& o, DBSCAN& d );
}

#endif // DBSCAN_VP_H
