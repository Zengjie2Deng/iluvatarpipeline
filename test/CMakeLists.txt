add_executable(pipeline1 Pipeline_OneStage.cpp)
target_link_libraries(pipeline1 ix_pipeline_help)

add_executable(pipeline2 Pipeline_TwoStage.cpp)
target_link_libraries(pipeline2 ix_pipeline_help)

add_executable(pipeline_multi Pipeline_MultiGPU.cpp ../src/apps/pipeline_multi_gpu.cpp)
target_link_libraries(pipeline_multi ix_pipeline_help)
