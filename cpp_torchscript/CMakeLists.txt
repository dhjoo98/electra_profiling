cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(importing_ts)
#set(Torch_DIR "libtorch/share/cmake/Torch") #this causes problems 
find_package(Torch REQUIRED)

add_executable(ts-infer ts-infer.cpp)
target_link_libraries(ts-infer "${TORCH_LIBRARIES}")
set_property(TARGET ts-infer PROPERTY CXX_STANDARD 14)
