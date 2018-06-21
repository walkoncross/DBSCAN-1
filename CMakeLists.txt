project(DBSCAN)

CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

#SET(CMAKE_C_FLAGS "-std=gnu99 -fPIC -pedantic -fopenmp -mtune=native")
#SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -fPIC -pedantic -fopenmp -mtune=native")

# remove option "-pedantic" to avoid compiler errors
SET(CMAKE_C_FLAGS "-std=gnu99 -fPIC -fopenmp -mtune=native")
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -fPIC -fopenmp -mtune=native")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME ON)

# find_package(Boost REQUIRED COMPONENTS program_options python3 numpy3)
# remove boost.numpy because it's now included in boost.python
find_package(Boost REQUIRED COMPONENTS program_options python3)

## for python3.5
#SET(PYTHON_INCLUDE_PATH /usr/include/python3.5)
#SET(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython3.5m.so.1)
#SET(PYTHON_EXECUTABLE /usr/bin/python3)
#SET(PYTHON_INCLUDE_DIR  /usr/include/python3.5)
#SET(PYTHONLIBS_FOUND True)
#message( "Using boost libraries: ${Boost_FOUND}")

## for python2.7
SET(PYTHON_INCLUDE_PATH /usr/include/python2.7)
SET(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython2.7.so.1)
SET(PYTHON_EXECUTABLE /usr/bin/python)
SET(PYTHON_INCLUDE_DIR  /usr/include/python2.7)
SET(PYTHONLIBS_FOUND True)
message( "Using boost libraries: ${Boost_FOUND}")

if (Boost_FOUND)
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    message(STATUS "Using boost include dirs: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Using boost library dirs: ${Boost_LIBRARY_DIRS}")
    message(STATUS "Using boost libraries: ${Boost_LIBRARIES}")
    message(STATUS "Using boost thread library: ${Boost_THREAD_LIBRARY}")
    link_directories(${Boost_LIBRARY_DIRS})
endif (Boost_FOUND)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_CXX_FLAGS_DEBUG "-g3" CACHE STRING "Debug CXX compiler flags" FORCE)
      set(CMAKE_C_FLAGS_DEBUG "-g3" CACHE STRING "Debug C compiler flags" FORCE)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

#add_subdirectory(gflags)
add_subdirectory(glog)
add_subdirectory(googletest)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR} 
	${CMAKE_SOURCE_DIR}/eigen
)

add_definitions(-DBOOST_UBLAS_NDEBUG -DGFLAGS_NAMESPACE=google)

add_library(dbscan dbscan.cpp)

#add_executable(test_dbscan test_dbscan.cpp)
#target_link_libraries(test_dbscan dbscan)

if (PYTHONLIBS_FOUND AND Boost_FOUND)	
	include_directories( 
		${PYTHON_INCLUDE_PATH} 
	)
	         
	ADD_LIBRARY(pydbscan SHARED PyDBSCAN.cpp)
	target_link_libraries(pydbscan dbscan glog boost_python3 boost_numpy3)

	SET_TARGET_PROPERTIES(pydbscan
		PROPERTIES
		    MAJOR_VERSION 1
		    MINOR_VERSION 0
		        LIBRARY_OUTPUT_NAME pydbscan
		        PREFIX ""
	)

endif(PYTHONLIBS_FOUND AND Boost_FOUND)

# add_executable(vptree main_vptree.cpp)

# add_executable(dset_test dataset_test.cpp)
# add_executable(dbscan_vp test_dbscan_vp.cpp)
# add_executable(dbscan_vp_cosine test_dbscan_vp_cosine.cpp)

enable_testing()

MESSAGE("CMAKE_C_FLAGS ${CMAKE_C_FLAGS}")
MESSAGE("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

find_package(CUDA)

if(${CUDA_FOUND})

MESSAGE("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")

endif(${CUDA_FOUND})

# if (${CUDA_FOUND})
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --x cu")
# endif()

add_subdirectory(tests)
add_subdirectory(tools)