INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/glog/src/)
INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/googletest/googletest/include )
INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/googletest/googlemock/include )

SET(MAIN_TEST_CPP ${PROJECT_SOURCE_DIR}/tests/main_test.cpp)
SET(MAIN_TEST_GPU_CPP ${PROJECT_SOURCE_DIR}/tests/main_test_gpu.cpp)

# Common function for test environment set up
FUNCTION( setupTests module sources libraries )
    ADD_DEFINITIONS( -DSOURCE_DIR=\"${CMAKE_SOURCE_DIR}/\" )
    ADD_DEFINITIONS( -DBIN_DIR=\"${CMAKE_BINARY_DIR}/\" )
    ADD_DEFINITIONS( -DCURRENT_TEST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\" )
    ADD_DEFINITIONS( -DCURRENT_BIN_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\" )

    INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/.. )

    SET( exeName test${module} )
    ADD_EXECUTABLE( ${exeName} ${sources} )

    SET_TARGET_PROPERTIES (${exeName} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    LINK_DIRECTORIES( ${PROJECT_BINARY_DIR}/${module} )

    TARGET_LINK_LIBRARIES( ${exeName} ${libraries} glog gtest gmock pthread )

ADD_TEST( ${exeName} ${CMAKE_BINARY_DIR}/tests/${exeName} --gtest_output=xml:${CMAKE_BINARY_DIR}/junit/${testResultName})
set_tests_properties(${exeName} PROPERTIES ENVIRONMENT "GLIBCXX_FORCE_NEW=1")
ENDFUNCTION( setupTests )

# Common function for test environment set up
FUNCTION( setupNvTests module sources libraries )
    ADD_DEFINITIONS( -DSOURCE_DIR=\"${CMAKE_SOURCE_DIR}/\" )
    ADD_DEFINITIONS( -DBIN_DIR=\"${CMAKE_BINARY_DIR}/\" )
    ADD_DEFINITIONS( -DCURRENT_TEST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\" )
    ADD_DEFINITIONS( -DCURRENT_BIN_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\" )

    INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/.. )
    INCLUDE_DIRECTORIES( SYSTEM ${CUDA_INCLUDE_DIRS})

    SET( exeName test${module} )
    CUDA_ADD_EXECUTABLE( ${exeName} ${sources} )

    SET_TARGET_PROPERTIES (${exeName} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    LINK_DIRECTORIES( ${PROJECT_BINARY_DIR}/${module} )

    TARGET_LINK_LIBRARIES( ${exeName} ${libraries} glog gtest gmock pthread )

ADD_TEST( ${exeName} ${CMAKE_BINARY_DIR}/tests/${exeName} --gtest_output=xml:${CMAKE_BINARY_DIR}/junit/${testResultName})
set_tests_properties(${exeName} PROPERTIES ENVIRONMENT "GLIBCXX_FORCE_NEW=1")
ENDFUNCTION( setupNvTests )

SET( SOURCES
    ${MAIN_TEST_CPP}
    test_vptree.cpp
    test_dbscan_vp.cpp
    test_dbscan_vp_cosine.cpp
    test_dataset.cpp
)

SET( LIBRARIES )

setupTests( "dbscan" "${SOURCES}" "${LIBRARIES}" )

SET( SOURCES
    ${MAIN_TEST_CPP}
    test_benchmark.cpp
)

SET( LIBRARIES )

setupTests( "benchmark" "${SOURCES}" "${LIBRARIES}" )

if (${CUDA_FOUND})

SET( SOURCES
    ${MAIN_TEST_GPU_CPP}
    ../nv/g_dbscan.cu
    ../nv/g_dbscan.cpp
    test_gpudbscan.cpp
)

SET( LIBRARIES ${CUDA_LIBRARIES} cuda)

setupNvTests( "gpu" "${SOURCES}" "${LIBRARIES}" )

endif(${CUDA_FOUND})
