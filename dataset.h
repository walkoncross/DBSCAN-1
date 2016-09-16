#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <iostream>
#include <fstream>

#include <unordered_map>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/noncopyable.hpp>

#include <Eigen/Dense>

namespace clustering {
class Dataset : private boost::noncopyable {
public:
    typedef boost::shared_ptr< Dataset > Ptr;
    typedef std::vector< Eigen::VectorXf > DataContainer;
    typedef Eigen::VectorXf LabelsContainer;

    static Ptr create()
    {
        return boost::make_shared< Dataset >();
    }

    Dataset()
    {
    }

    void gen_cluster_data( size_t features_num, size_t elements_num )
    {
        _data.clear();
        _labels.resize( elements_num );

        for ( size_t i = 0; i < elements_num; ++i ) {
            Eigen::VectorXf col_vector( features_num );
            for ( size_t j = 0; j < features_num; ++j ) {
                col_vector( j ) = ( -1.0 + rand() * ( 2.0 ) / RAND_MAX );
            }
            _data.emplace_back( col_vector );
        }
    }

    bool load_csv( const std::string& csv_file_path )
    {
        std::ifstream in( csv_file_path );

        if ( !in.is_open() ) {
            std::cout << "Not opened " << csv_file_path << std::endl;
            return false;
        }

        std::string line;

        std::vector< float > row_cache;
        std::vector< float > labels_cache;
        std::unordered_map< std::string, size_t > known_labels;
        size_t label_idx = 0;

        while ( std::getline( in, line ) ) {
            if ( !line.size() ) {
                continue;
            }

            row_cache.clear();

            const char* ptr = line.c_str();
            size_t len = line.length();

            const char* start = ptr;
            for ( size_t i = 0; i < len; ++i ) {

                if ( ptr[i] == ',' ) {
                    row_cache.push_back( std::atof( start ) );
                    start = ptr + i + 1;
                }
            }

            const std::string label_str( start );

            auto r = known_labels.find( start );

            size_t found_label = label_idx;

            if ( r == known_labels.end() ) {
                known_labels.insert( std::make_pair( label_str, label_idx ) );
                std::cout << "Found new label " << label_str << " " << found_label << std::endl;
                ++label_idx;
            } else {
                found_label = r->second;
            }

            labels_cache.push_back( float( found_label ) );

            Eigen::VectorXf col_vector( row_cache.size() );
            for ( size_t i = 0; i < row_cache.size(); ++i ) {
                col_vector( i ) = row_cache[i];
            }

            _data.emplace_back( col_vector );
        }

        in.close();

        _labels.resize( labels_cache.size() );
        for ( size_t i = 0; i < labels_cache.size(); ++i ) {
            _labels( i ) = labels_cache[i];
        }

        return true;
    }

    DataContainer& data()
    {
        return _data;
    }

    const LabelsContainer& get_labels() const
    {
        return _labels;
    }

private:
    DataContainer _data;
    LabelsContainer _labels;
};
}

#endif // DATASET_H
