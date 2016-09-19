#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <omp.h>

#include "dbscan_vp.h"
#include "dataset.h"

using namespace clustering;

namespace {
static const std::string CURRENT_TDIR( CURRENT_TEST_DIR );

static const size_t NUM_MEASURES = 20;
static const size_t NUM_FEATURES = 50;
static const size_t STEP_LEN = 10000;
static const size_t NUM_STEPS = 100;
}

TEST( DBSCAN_VP, Iris )
{
    std::vector< size_t > MEASURE_SIZES_VECTOR;
    for ( size_t i = STEP_LEN; i < STEP_LEN * NUM_STEPS; i += STEP_LEN ) {
        MEASURE_SIZES_VECTOR.push_back( i );
    }

    Eigen::VectorXd means( MEASURE_SIZES_VECTOR.size() );

    for ( size_t j = 0; j < MEASURE_SIZES_VECTOR.size(); ++j ) {
        const size_t vsz = MEASURE_SIZES_VECTOR[j];

        LOG( INFO ) << "Data matrix size " << NUM_FEATURES << "x" << vsz;
        Dataset::Ptr dset = Dataset::create();
        dset->gen_cluster_data( NUM_FEATURES, vsz );

        Eigen::VectorXd measures( NUM_MEASURES );

        for ( size_t i = 0; i < NUM_MEASURES; ++i ) {
            DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( 0.01, 5, 1 );
            const double start = omp_get_wtime();
            dbs->fit( dset );
            const double end = omp_get_wtime();

            measures( i ) = end - start;

            VLOG( 3 ) << "Measure " << ( i + 1 ) << " time = " << measures( i ) << " seconds";
        }

        LOG( INFO ) << "min = " << measures.minCoeff() << " max = " << measures.maxCoeff() << " mean = " << measures.mean();
        means( j ) = measures.mean();
    }

    static const std::string stat_file_name = "stat.csv";

    std::ofstream stat_file;
    stat_file.open( stat_file_name.c_str(), std::ios::out | std::ios::trunc );

    stat_file << "data size,seconds" << std::endl;

    for ( size_t j = 0; j < MEASURE_SIZES_VECTOR.size(); ++j ) {
        stat_file << MEASURE_SIZES_VECTOR[j] << "," << means( j ) << std::endl;
    }

    stat_file.close();

    LOG( INFO ) << "Statistics written to " << stat_file_name;
}