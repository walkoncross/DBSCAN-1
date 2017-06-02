#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "dbscan.h"
#include "dbscan_vp.h"

using namespace boost::python;
using namespace clustering;

namespace p = boost::python;
namespace np = boost::python::numpy;

struct ublas_matrix_to_python {
    static PyObject* convert( DBSCAN::ClusterData const& C )
    {
        PyObject* result = PyList_New( C.size1() );

        for ( size_t i = 0; i < C.size1(); ++i ) {
            PyObject* l = PyList_New( C.size2() );

            for ( size_t j = 0; j < C.size2(); ++j ) {
                PyList_SetItem( l, j, PyFloat_FromDouble( C( i, j ) ) );
            }

            PyList_SetItem( result, i, l );
        }

        return result;
    }
};

class PyDBSCAN : public DBSCAN {
public:
    PyDBSCAN()
        : DBSCAN()
    {
    }

    void pyfit( boost::python::list& pylist )
    {
        auto num_samples = boost::python::len( pylist );

        if ( num_samples == 0 ) {
            return;
        }

        boost::python::extract< boost::python::list > first_elem( pylist[0] );

        auto num_features = boost::python::len( first_elem );

        DBSCAN::ClusterData C( num_samples, num_features );

        for ( int i = 0; i < num_samples; ++i ) {
            boost::python::list sublist = boost::python::extract< boost::python::list >( pylist[i] );

            for ( int j = 0; j < num_features; ++j ) {
                C( i, j ) = boost::python::extract< double >( sublist[j] );
            }
        }

        fit( C );
    }

    void pywfit( boost::python::list& pylist, boost::python::list& pyweights )
    {
        auto num_samples = boost::python::len( pylist );

        if ( num_samples == 0 ) {
            return;
        }

        boost::python::extract< boost::python::list > first_elem( pylist[0] );

        auto num_features = boost::python::len( first_elem );

        DBSCAN::ClusterData C( num_samples, num_features );

        for ( int i = 0; i < num_samples; ++i ) {
            boost::python::list sublist = boost::python::extract< boost::python::list >( pylist[i] );

            for ( int j = 0; j < num_features; ++j ) {
                C( i, j ) = boost::python::extract< double >( sublist[j] );
            }
        }

        auto num_weights = boost::python::len( pyweights );

        DBSCAN::FeaturesWeights W( num_weights );

        for ( int i = 0; i < num_weights; ++i ) {
            W( i ) = boost::python::extract< double >( pyweights[i] );
        }

        wfit( C, W );
    }

    boost::python::list pyget_labels()
    {
        boost::python::list list;

        for ( const auto& l : get_labels() ) {
            list.append( l );
        }

        return list;
    }
};

class npDataset : public Dataset {
public:
    typedef boost::shared_ptr< npDataset > Ptr;

    npDataset()
        : Dataset()
    {
    }

    virtual ~npDataset()
    {
    }

    bool load_ndarray( const np::ndarray& in )
    {
        m_rows = in.shape( 0 );
        m_cols = in.shape( 1 );

        for ( size_t i = 0; i < m_rows; ++i ) {
            Eigen::VectorXf col_vector( m_cols );
            for ( size_t j = 0; j < m_cols; ++j ) {
                col_vector( j ) = p::extract< float >( in[i][j] );
            }

            _data.emplace_back( col_vector );
        }

        return true;
    }
};

class PyDBSCANvp {
public:
    PyDBSCANvp( const np::ndarray& d )
        : m_rows( 0 )
    {
        npDataset::Ptr np_dset = boost::make_shared< npDataset >();
        np_dset->load_ndarray( d );

        m_dbs = boost::make_shared< DBSCAN_VP >( np_dset );
        m_dbs->fit();
        m_rows = np_dset->rows();
    }

    const np::ndarray predict( double eps, size_t min_elems )
    {
        np::dtype dtype = np::dtype::get_builtin< int32_t >();
        p::tuple shape = p::make_tuple( m_rows );

        np::ndarray rnd = np::zeros( shape, dtype );

        m_dbs->predict( eps, min_elems );

        const DBSCAN_VP::Labels& labels = m_dbs->get_labels();

        for ( size_t i = 0; i < labels.size(); ++i ) {
            rnd[i] = labels[i];
        }

        return rnd;
    }

    const np::ndarray predict_eps( size_t k )
    {
        np::dtype dtype = np::dtype::get_builtin< float >();
        p::tuple shape = p::make_tuple( m_rows );

        np::ndarray rnd = np::zeros( shape, dtype );

        const std::vector< double > r = m_dbs->predict_eps( k );

        for ( size_t i = 0; i < r.size(); ++i ) {
            rnd[i] = r[i];
        }
        return rnd;
    }

private:
    size_t m_rows;
    DBSCAN_VP::Ptr m_dbs;
};

BOOST_PYTHON_MODULE( pydbscan )
{
    np::initialize();

    def( "gen_cluster_data", &PyDBSCAN::gen_cluster_data );
    //def( "send_stat", & python_send_stat );

    class_< PyDBSCAN >( "DBSCAN" )
        .def( "init", &PyDBSCAN::init )
        .def( "fit", &PyDBSCAN::pyfit )
        .def( "wfit", &PyDBSCAN::pywfit )
        .def( "get_labels", &PyDBSCAN::pyget_labels );

    class_< PyDBSCANvp >( "DBSCANvp", p::init< np::ndarray >() )
        .def( "predict", &PyDBSCANvp::predict )
        .def( "predict_eps", &PyDBSCANvp::predict_eps );

    to_python_converter< DBSCAN::ClusterData, ublas_matrix_to_python, false >();
}
