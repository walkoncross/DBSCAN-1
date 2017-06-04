#include <iostream>
#include "nv/CudaCtx.h"

int main(int argc, char const *argv[])
{
	CudaCtx CUctx;

	std::cout << "device count " << CUctx.getDevCount() << std::endl;
	std::cout << "device name " << CUctx.getDevName() << std::endl;
	std::cout << "device memory " << CUctx.getDevMemory() << " bytes" << std::endl;

	return 0;
}
