cmake_minimum_required (VERSION 3.16.3)
# option(BUILD_TESTS ON "Build unit tests")
project(llama2cpp
        VERSION 0.1.0
        LANGUAGES CXX 
        DESCRIPTION "llama2 inference in C++"
        )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

option(BUILD_EXAMPLES OFF "Build examples")

include_directories(include)

file(GLOB HDR_FILES ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/*.hpp)

add_library(llama2cpp_core SHARED)
target_sources(llama2cpp_core
        PUBLIC
                ${HDR_FILES}
)
target_include_directories(llama2cpp_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/llama2cpp_core>
  $<INSTALL_INTERFACE:include/llama2cpp_core>  # <prefix>/include/mylib
)
set_target_properties(llama2cpp_core PROPERTIES LINKER_LANGUAGE CXX)

add_executable(
    run_llama src/main.cpp
)
target_link_libraries(run_llama PRIVATE llama2cpp_core)

if(BUILD_TESTS)
    # enable_testing() # TODO add gtest
    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE llama2cpp_core)

    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE llama2cpp_core)
endif()