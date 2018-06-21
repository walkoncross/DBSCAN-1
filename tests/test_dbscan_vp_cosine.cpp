#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>
#include <omp.h>

#include <dataset.h>
#include <Eigen/Dense>

#include "logging.h"

#include "dbscan_vp_cosine.h"

namespace
{
static const std::string CURRENT_TDIR(CURRENT_TEST_DIR);
}

using namespace clustering;

TEST(DBSCAN_VP_COSINE, TwoClusters)
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv(CURRENT_TDIR + "/csv/vptree01.csv");

    dset->L2_normalize(); // features must be L2-normalized before using cosine distance

    DBSCAN_VP_COSINE::Ptr dbs = boost::make_shared<DBSCAN_VP_COSINE>(dset);
    dbs->fit();
    dbs->predict(0.01, 5);

    const DBSCAN_VP_COSINE::Labels &l = dbs->get_labels();

    for (size_t i = 0; i < l.size(); ++i)
    {
        LOG(INFO) << "Element = " << i << " cluster = " << l[i];
        if (i < 5)
        {
            EXPECT_EQ(l[i], 0);
        }
        else
        {
            EXPECT_EQ(l[i], 1);
        }
    }
}

TEST(DBSCAN_VP_COSINE, OneCluster)
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv(CURRENT_TDIR + "/csv/vptree02.csv");

    dset->L2_normalize(); // features must be L2-normalized before using cosine distance

    DBSCAN_VP_COSINE::Ptr dbs = boost::make_shared<DBSCAN_VP_COSINE>(dset);
    dbs->fit();
    dbs->predict(0.01, 5);

    const DBSCAN_VP_COSINE::Labels &l = dbs->get_labels();

    for (size_t i = 0; i < l.size(); ++i)
    {
        LOG(INFO) << "Element = " << i << " cluster = " << l[i];
        if (i < 6)
        {
            EXPECT_EQ(l[i], 0);
        }
        else
        {
            EXPECT_EQ(l[i], -1);
        }
    }
}

TEST(DBSCAN_VP_COSINE, NoClusters)
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv(CURRENT_TDIR + "/csv/vptree03.csv");

    dset->L2_normalize(); // features must be L2-normalized before using cosine distance

    DBSCAN_VP_COSINE::Ptr dbs = boost::make_shared<DBSCAN_VP_COSINE>(dset);
    dbs->fit();
    dbs->predict(0.01, 2);

    const DBSCAN_VP_COSINE::Labels &l = dbs->get_labels();

    for (size_t i = 0; i < l.size(); ++i)
    {
        LOG(INFO) << "Element = " << i << " cluster = " << l[i];
        EXPECT_EQ(l[i], -1);
    }
}

TEST(DBSCAN_VP_COSINE, Iris)
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv(CURRENT_TDIR + "/csv/iris.data.txt");

    DBSCAN_VP_COSINE::Ptr dbs = boost::make_shared<DBSCAN_VP_COSINE>(dset);

    dset->L2_normalize(); // features must be L2-normalized before using cosine distance

    dbs->fit();
    dbs->predict(0.4, 5);

    const DBSCAN_VP_COSINE::Labels &l = dbs->get_labels();

    for (size_t i = 0; i < l.size(); ++i)
    {
        LOG(INFO) << "Element = " << i << " cluster = " << l[i];
    }
}

TEST(DBSCAN_VP_COSINE, IrisAnalyze)
{
    Dataset::Ptr dset = Dataset::create();
    dset->load_csv(CURRENT_TDIR + "/csv/iris.data.txt");

    dset->L2_normalize(); // features must be L2-normalized before using cosine distance

    DBSCAN_VP_COSINE::Ptr dbs = boost::make_shared<DBSCAN_VP_COSINE>(dset);

    dbs->fit();
    const auto r = dbs->predict_eps(3u);

    for (size_t i = 0; i < r.size(); ++i)
    {
        std::cout << (i + 1) << "," << r[i] << std::endl;
    }
}
