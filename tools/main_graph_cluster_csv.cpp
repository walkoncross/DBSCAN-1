#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

#include "glog/logging.h"

#include "dbscan_graph.h"

namespace po = boost::program_options;
using namespace clustering;

int
main( int argc, char const* argv[] )
{
    std::string file_in;
    double eps = 0.1;
    size_t numpts = 3;

    po::options_description option_desc( "DBSCAN clusterizer" );

    option_desc.add_options()( "help", "Display help" )(
        "in,i",
        po::value< std::string >( &file_in )->required(),
        "Input file CSV format" )(
        "eps,e", po::value< double >( &eps )->default_value( 0.1 ), "Epsilon param" )(
        "numpts,n",
        po::value< size_t >( &numpts )->default_value( 3u ),
        "Min number of points in cluster" );

    po::variables_map options;

    po::store( po::command_line_parser( argc, argv ).options( option_desc ).run(),
               options );

    if ( options.empty() or options.count( "help" ) ) {
        std::cout << option_desc << std::endl;
        return 0;
    }

    po::notify( options );

    Dataset::Ptr dset = Dataset::create();
    LOG( INFO ) << "Loading dataset from " << file_in;
    dset->load_csv( file_in );

    DBSCAN_GRAPH gdbs( dset );

    gdbs.fit( eps, numpts );

    LOG( INFO ) << "Fit time " << gdbs.get_fit_time() << " seconds";

    const int32_t num_clusters = gdbs.predict();

    LOG( INFO ) << "Predict time " << gdbs.get_predict_time() << " seconds";

    // const DBSCAN_GRAPH::Labels& l = gdbs.get_labels();

    // for ( size_t i = 0; i < l.size(); ++i ) {
    //     std::cout << l[i] << std::endl;
    // }

    LOG( INFO ) << "Num clusters " << num_clusters;

    return 0;
}
