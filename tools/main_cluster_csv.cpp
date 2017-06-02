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
    std::string file_out;
    std::string file_format;
    double eps = 0.1;
    size_t numpts = 3;

    po::options_description option_desc( "DBSCAN clusterizer" );

    option_desc.add_options()( "help", "Display help" )(
        "in,i", po::value< std::string >( &file_in )->required(), "Input file (CSV format)" )(
        "out,o", po::value< std::string >( &file_out )->required(), "Output file" )(
        "type,t", po::value< std::string >( &file_format )->default_value( "csv" ), "Output file format" )(
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

    const uint32_t num_clusters = dbs->predict( eps, numpts );

    LOG( INFO ) << "Predict time " << dbs->get_predict_time() << " seconds";

    const DBSCAN_VP::Labels& l = dbs->get_labels();

    std::ofstream fl( file_out.c_str(), std::ios::out | std::ios::trunc );

    if ( file_format == "txt" ) {

        for ( ssize_t cl_id = 0; cl_id < num_clusters; ++cl_id ) {
            fl << "Cluster " << cl_id << std::endl;
            size_t cluster_total = 0;
            for ( size_t i = 0; i < l.size(); ++i ) {
                if ( l[i] == cl_id ) {
                    fl << "\t" << i << "\t" << dset->get_label( i ) << std::endl;
                    ++cluster_total;
                }
            }
            if ( cluster_total ) {
                fl << "Total " << cluster_total << std::endl;
            }
        }

        fl << "Outliers (-1)" << std::endl;

        size_t cluster_total = 0;
        for ( size_t i = 0; i < l.size(); ++i ) {
            if ( l[i] == -1 ) {
                fl << "\t" << i << "\t" << dset->get_label( i ) << std::endl;
                ++cluster_total;
            }
        }

        fl << "Total " << cluster_total << std::endl;
    } else if ( file_format == "csv" ) {
        for ( size_t i = 0; i < l.size(); ++i ) {
            fl << l[i] << std::endl;
        }
    }

    fl.close();

    // std::cout << "id,cluster_id" << std::endl;

    // for ( size_t i = 0; i < l.size(); ++i ) {
    //     std::cout << i << "," << l[i] << std::endl;
    // }

    LOG( INFO ) << "Num clusters " << num_clusters;

    return 0;
}