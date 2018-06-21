#include "gtest/gtest.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

#include <dataset.h>

#include "glog/logging.h"

#include "nv/g_dbscan.h"

#include "dbscan_vp.h"
#include "dbscan_vp_cosine.h"

namespace
{
static const std::string CURRENT_TDIR(CURRENT_TEST_DIR);
}

using namespace clustering;

TEST(GDBSCAN, TwoClusters)
{
  Dataset::Ptr dset = Dataset::create();
  dset->load_csv(CURRENT_TDIR + "/csv/vptree01.csv");

  GDBSCAN::Ptr dbs = boost::make_shared<GDBSCAN>(dset);
  dbs->fit(0.01, 5);
  dbs->predict();

  const GDBSCAN::Labels &l = dbs->get_labels();

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

TEST(GDBSCAN, OneCluster)
{
  Dataset::Ptr dset = Dataset::create();
  dset->load_csv(CURRENT_TDIR + "/csv/vptree02.csv");

  GDBSCAN::Ptr dbs = boost::make_shared<GDBSCAN>(dset);
  dbs->fit(0.01, 5);
  dbs->predict();

  const GDBSCAN::Labels &l = dbs->get_labels();

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

TEST(GDBSCAN, NoClusters)
{
  Dataset::Ptr dset = Dataset::create();
  dset->load_csv(CURRENT_TDIR + "/csv/vptree03.csv");

  GDBSCAN::Ptr dbs = boost::make_shared<GDBSCAN>(dset);
  dbs->fit(0.01, 2);
  dbs->predict();

  const GDBSCAN::Labels &l = dbs->get_labels();

  for (size_t i = 0; i < l.size(); ++i)
  {
    LOG(INFO) << "Element = " << i << " cluster = " << l[i];
    EXPECT_EQ(l[i], -1);
  }
}

TEST(GDBSCAN, Iris)
{
  Dataset::Ptr dset = Dataset::create();
  dset->load_csv(CURRENT_TDIR + "/csv/iris.data.txt");

  GDBSCAN::Ptr dbs = boost::make_shared<GDBSCAN>(dset);

  dbs->fit(0.4, 5);
  dbs->predict();

  const GDBSCAN::Labels &l = dbs->get_labels();

  for (size_t i = 0; i < l.size(); ++i)
  {
    LOG(INFO) << "Element = " << i << " cluster = " << l[i];
  }
}

TEST(GDBSCAN, vsDBSCANvp)
{
  const float eps = 0.3;
  const size_t num_pts = 10;

  Dataset::Ptr dset = Dataset::create();
  dset->load_csv(CURRENT_TDIR + "/csv/gpu1000.csv");

  GDBSCAN::Ptr gdbs = boost::make_shared<GDBSCAN>(dset);

  gdbs->fit(eps, num_pts);
  int32_t numcl = gdbs->predict();
  LOG(INFO) << "GPU numcl " << numcl << " fit " << gdbs->get_fit_time()
            << " predict " << gdbs->get_predict_time();

  DBSCAN_VP::Ptr dbs = boost::make_shared<DBSCAN_VP>(dset);

  dbs->fit();
  numcl = dbs->predict(eps, num_pts);

  LOG(INFO) << "CPU numcl " << numcl << " fit " << dbs->get_fit_time()
            << " predict " << dbs->get_predict_time();
}

TEST(GDBSCAN, vsDBSCANvp_cosine)
{
  const float eps = 0.3;
  const size_t num_pts = 10;

  Dataset::Ptr dset = Dataset::create();
  dset->load_csv(CURRENT_TDIR + "/csv/gpu1000.csv");

  dset->L2_normalize(); // features must be L2-normalized before using cosine distance

  GDBSCAN::Ptr gdbs = boost::make_shared<GDBSCAN>(dset);

  gdbs->fit(eps, num_pts, 1);
  int32_t numcl = gdbs->predict();
  LOG(INFO) << "GPU numcl " << numcl << " fit " << gdbs->get_fit_time()
            << " predict " << gdbs->get_predict_time();

  DBSCAN_VP_COSINE::Ptr dbs = boost::make_shared<DBSCAN_VP_COSINE>(dset);

  dbs->fit();
  numcl = dbs->predict(eps, num_pts);

  LOG(INFO) << "CPU numcl " << numcl << " fit " << dbs->get_fit_time()
            << " predict " << dbs->get_predict_time();
}
