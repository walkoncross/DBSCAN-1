#include <iostream>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>

#include "glog/logging.h"

#include "dbscan_vp.h"

namespace po = boost::program_options;
using namespace clustering;

int main( int argc, char const* argv[] )
{
    std::string file_in;
    double eps = 0.1;
    size_t numpts = 3;

    po::options_description option_desc( "DBSCAN clusterizer" );

    option_desc.add_options()( "help", "Display help" )(
        "in,i", po::value< std::string >( &file_in )->required(), "Input file CSV format" )(
        "eps,e", po::value< double >( &eps )->default_value( 0.1 ), "Epsilon param" )(
        "numpts,n", po::value< size_t >( &numpts )->default_value( 3u ), "Min number of points in cluster" );

    po::variables_map options;

    po::store( po::command_line_parser( argc, argv ).options( option_desc ).run(), options );

    if ( options.empty() or options.count( "help" ) ) {
        std::cout << option_desc << std::endl;
        return 0;
    }

    po::notify( options );

    Dataset::Ptr dset = Dataset::create();
    LOG( INFO ) << "Loading dataset from " << file_in;
    dset->load_csv( file_in );

    DBSCAN_VP::Ptr dbs = boost::make_shared< DBSCAN_VP >( dset );

    dbs->fit();

    LOG( INFO ) << "Fit time " << dbs->get_fit_time() << " seconds";

    dbs->predict( 0.4, 5 );

    LOG( INFO ) << "Predict time " << dbs->get_predict_time() << " seconds";

    std::cout << "id,cluster_id" << std::endl;

    const DBSCAN_VP::Labels& l = dbs->get_labels();

    for ( size_t i = 0; i < l.size(); ++i ) {
        std::cout << i << "," << l[i] << std::endl;
    }

    return 0;
}