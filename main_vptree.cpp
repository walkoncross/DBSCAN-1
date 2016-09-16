#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>
#include <omp.h>

#include <Eigen/Dense>

#include "vptree.h"
#include "dataset.h"

using namespace clustering;

inline double dist( const Eigen::VectorXf& p1, const Eigen::VectorXf& p2 )
{
    return ( p1 - p2 ).norm();
}

typedef VPTREE< Eigen::VectorXf, dist > TTree;

int main( int argc, char const* argv[] )
{
    Dataset::Ptr dset = Dataset::create();
    std::cout << dset->load_csv( argv[1] ) << std::endl;

    double start = omp_get_wtime();
    TTree tree;
    tree.create( dset );
    double end = omp_get_wtime();
    std::cout << "creating tree took: " << end - start << " seconds" << std::endl;

    const Dataset::DataContainer& d = dset->data();

    std::cout << d.size() << std::endl;

    start = omp_get_wtime();
    //#pragma omp parallel for
    for ( size_t i = 0; i < std::min( size_t( 10 ), d.size() ); ++i ) {

        TTree::TNeighborsList nlist;

        std::cout << "Searching for " << i << std::endl;

        tree.search( d[i], 1.0, nlist );

        for ( size_t j = 0; j < nlist.size(); ++j ) {
            std::cout << nlist[j].first << " " << nlist[j].second << std::endl;
        }
    }
    end = omp_get_wtime();
    std::cout << "searching neighbors for all particles took: " << end - start << " seconds" << std::endl;

    return 0;
}
