INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/glog/src/)
INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/googletest/googletest/include/gtest )
INCLUDE_DIRECTORIES( SYSTEM ${PROJECT_SOURCE_DIR}/googletest/googlemock/include/gmock )

SET(MAIN_TEST_CPP ${PROJECT_SOURCE_DIR}/tests/main_test.cpp)

# Common function for test environment set up
FUNCTION( setupTests module sources libraries )
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
