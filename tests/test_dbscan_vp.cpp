#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>
#include <omp.h>

#include <dataset.h>
#include <Eigen/Dense>

#include "dbscan_vp.h"

using namespace clustering;

int main( int argc, char const* argv[] )
{
    double start = omp_get_wtime();
    Dataset::Ptr dset = Dataset::create();
    //dset->gen_cluster_data( 225, 100000 );
    dset->load_csv( argv[1] );
    double end = omp_get_wtime();

    std::cout << "data gen took: " << end - start << " seconds" << std::endl;

    start = omp_get_wtime();

    DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( 0.3, 5, 1 );
    dbs->fit( dset );

    end = omp_get_wtime();

    std::cout << "clustering took: " << end - start << " seconds" << std::endl;

    std::cout << "[ ";
    for ( const auto& l : dbs->get_labels() ) {
        std::cout << " " << l;
    }
    std::cout << " ] " << std::endl;

    const auto& labels = dset->get_labels();

    std::cout
        << "[ ";
    for ( ssize_t i = 0; i < labels.size(); ++i ) {
        std::cout << " " << uint32_t( labels[i] );
    }
    std::cout << " ] " << std::endl;

    // double start = omp_get_wtime();
    // VpTree< Eigen::VectorXf, dist > tree( dset->data() );
    // tree.create();
    // double end = omp_get_wtime();
    // std::cout << "creating tree took: " << end - start << " seconds" << std::endl;

    // const Dataset::DataContainer& d = dset->data();

    // std::cout << d.size() << std::endl;

    // start = omp_get_wtime();
    // //#pragma omp parallel for
    // for ( size_t i = 0; i < std::min( size_t( 10 ), d.size() ); ++i ) {
    //     std::vector< double > distances;
    //     std::vector< size_t > neighbors;

    //     std::cout << "Searching for " << i << std::endl;

    //     tree.search( d[i], 1.0, &neighbors, &distances );

    //     for ( size_t j = 0; j < distances.size(); ++j ) {
    //         std::cout << distances[j] << " " << neighbors[j] << std::endl;
    //     }
    // }
    // end = omp_get_wtime();
    // std::cout << "searching neighbors for all particles took: " << end - start << " seconds" << std::endl;

    // return 0;
}
