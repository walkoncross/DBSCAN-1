#include <cstdlib>
#include <ctime>

#include "nv/CudaCtx.h"
#include "gtest/gtest.h"

int
main(int argc, char** argv)
{
  CudaCtx ctx;

  std::random_device rd;
  std::mt19937 mt(rd());

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
