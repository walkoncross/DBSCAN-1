INCLUDE_DIRECTORIES( SYSTEM ${google-glog_BINARY_DIR} ${PROJECT_SOURCE_DIR}/glog/src/ )

add_executable(cluster_csv main_cluster_csv.cpp)
target_link_libraries(cluster_csv ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE} glog )

add_executable(graph_cluster_csv main_graph_cluster_csv.cpp)
target_link_libraries(graph_cluster_csv ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE} glog )

add_executable(knn_dist main_knn_dist.cpp)
target_link_libraries(knn_dist ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE} glog)

if (${CUDA_FOUND})

MESSAGE("${CUDA_FOUND} ${CUDA_LIBRARIES} ${CUDA_NVCC_FLAGS}")

cuda_add_executable(
    cudainfo
    cuda_info.cpp)

target_link_libraries(cudainfo ${CUDA_LIBRARIES} cuda)

cuda_add_executable(
    gcluster_csv
    ../nv/g_dbscan.cu
    ../nv/g_dbscan.cpp
    main_gcluster_csv.cpp)

target_link_libraries(gcluster_csv ${CUDA_LIBRARIES} cuda boost_program_options glog)

endif(${CUDA_FOUND})
