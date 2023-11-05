cmake_minimum_required(VERSION 3.26)

project(keepsake_test LANGUAGES CXX C)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(TBB CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)  
pkg_check_modules(LuaJIT REQUIRED IMPORTED_TARGET luajit)

set(KS_SRC
  ../assertion.h
  ../assertion.cpp
  ../config.h
  ../config.cpp
  ../keyframe.h
  ../keyframe.cpp
  ../parallel.h
  ../parallel.cpp
  ../image_util.h
  ../image_util.cpp
  ../yui/sol/config.hpp
  ../yui/sol/forward.hpp
  ../yui/sol/sol.hpp    
  ../yui/yui.h
  ../yui/yui.cpp  
)

set(KS_TEST_SRC
  main.cpp
)

set(COMMON_INCLUDE_DIRS
  ../yui  
  deps/cxxopts/include
  deps/pcg-cpp/include
  deps/stb
  deps/tinyexr
  deps/tomlplusplus
)

set(COMMON_COMPILE_DEFS
  -DPPK_ASSERT_ENABLED=1
  -DDATA_DIR=\"${CMAKE_SOURCE_DIR}/data/\"
)

set(CPP_LIBS
  Eigen3::Eigen
  TBB::tbb
  PkgConfig::LuaJIT
)

add_executable(keepsake_test)
target_sources(keepsake_test PRIVATE ${KS_SRC} ${KS_TEST_SRC})
target_compile_features(keepsake_test PRIVATE cxx_std_20)
target_include_directories(keepsake_test PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(keepsake_test PRIVATE ${CPP_LIBS})
set_target_properties(keepsake_test PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)