#include <dataset.h>

using namespace clustering;

int main( int argc, char const* argv[] )
{
    Dataset::Ptr dset = Dataset::create();

    std::cout << dset->load_csv( argv[1] ) << std::endl;

    return 0;
}