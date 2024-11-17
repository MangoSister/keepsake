cmake_minimum_required(VERSION 3.23)

project(keepsake_test LANGUAGES CXX C)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE_MODULE_PATH = " "${CMAKE_MODULE_PATH}")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(TBB CONFIG REQUIRED)
find_package(embree 3.0 REQUIRED)
find_package(lz4 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)  
pkg_check_modules(LuaJIT REQUIRED IMPORTED_TARGET luajit)

find_package(Vulkan)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)
find_package(VulkanUtilityLibraries CONFIG REQUIRED)
find_package(Freetype REQUIRED)

# Slang vcpkg+CMake setup from: https://github.com/Ipotrick/Daxa...
find_library(SLANG_LIB slang)
message("SLANG_LIB = " "${SLANG_LIB}")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  find_file(SLANG_DLL slang.dll PATH_SUFFIXES bin)
  find_file(SLANG_GLSLANG_DLL slang-glslang.dll PATH_SUFFIXES bin)
  message("SLANG_DLL = " "${SLANG_DLL}")
  message("SLANG_GLSLANG_DLL = " "${SLANG_GLSLANG_DLL}")
endif()

set(KS_SRC
  ../assertion.h
  ../assertion.cpp
  ../maths.h
  ../aabb.h
  ../barray.h
  ../bsdf.h
  ../bsdf.cpp
  ../principled_bsdf.h
  ../principled_bsdf.cpp
  ../camera.h
  ../camera.cpp
  ../config.h
  ../config.cpp
  ../distrib.h
  ../distrib.cpp
  ../embree_util.h
  ../embree_util.cpp
  ../file_util.h
  ../geometry.h
  ../geometry.cpp
  ../hash.h
  ../keyframe.h
  ../keyframe.cpp
  ../light.h
  ../light.cpp    
  ../log_util.h
  ../log_util.cpp
  ../memory_util.h
  ../memory_util.cpp
  ../material.h
  ../material.cpp
  ../mesh_asset.h
  ../mesh_asset.cpp  
  ../nee.h
  ../nee.cpp
  ../normal_map.h
  ../normal_map.cpp
  ../opacity_map.h
  ../opacity_map.cpp
  ../parallel.h
  ../parallel.cpp
  ../render_target.h
  ../render_target.cpp
  ../tonemap.h
  ../tonemap.cpp
  ../scene.h
  ../scene.cpp  
  ../subsurface.h
  ../subsurface.cpp
  ../texture.h
  ../texture.cpp    
  ../ray.h
  ../ray.cpp  
  ../rng.h
  ../rng.cpp
  ../image_util.h
  ../image_util.cpp
  ../sobol.h
  ../sobol.cpp
  ../sat.h
  ../small_pt.h
  ../small_pt.cpp
  ../colormap.h
  ../colormap.cpp
  ../yui/sol/config.hpp
  ../yui/sol/forward.hpp
  ../yui/sol/sol.hpp    
  ../yui/yui.h
  ../yui/yui.cpp  
    #
    ../gpu/ksvk.h
    ../gpu/ksvk.cpp
    ../gpu/scene.h
    ../gpu/scene.cpp   
    ../gpu/small_pt.h
    ../gpu/small_pt.cpp   
)

set(IMGUI_SRC
  ../deps/imgui/imconfig.h
  ../deps/imgui/imgui.h
  ../deps/imgui/imgui.cpp
  ../deps/imgui/imgui_demo.cpp
  ../deps/imgui/imgui_draw.cpp
  ../deps/imgui/imgui_internal.h
  ../deps/imgui/imgui_tables.cpp
  ../deps/imgui/imgui_widgets.cpp
  ../deps/imgui/backends/imgui_impl_glfw.h
  ../deps/imgui/backends/imgui_impl_glfw.cpp
  ../deps/imgui/backends/imgui_impl_vulkan.h
  ../deps/imgui/backends/imgui_impl_vulkan.cpp
  ../deps/imgui/misc/freetype/imgui_freetype.h
  ../deps/imgui/misc/freetype/imgui_freetype.cpp
)

set(KS_TEST_SRC
  main.cpp
)

set(COMMON_INCLUDE_DIRS
  ../yui
  ../deps/cxxopts/include
  ../deps/stb
  ../deps/tinyexr
  ../deps/tinyobjloader  
  ../deps/tinygltf
  ../deps/tinyply 
  ../deps/tomlplusplus
  ../deps/imgui/
  ../deps/stduuid/include
  ../deps/spdlog/include
  ../deps/
)

set(COMMON_COMPILE_DEFS
  -DVK_NO_PROTOTYPES
  -DPPK_ASSERT_ENABLED=1
  -DDATA_DIR=\"${CMAKE_SOURCE_DIR}/data/\"
  -DKS_SHADER_DIR=\"${CMAKE_SOURCE_DIR}/../gpu/shaders\"
)

set(CPP_LIBS
  Eigen3::Eigen
  TBB::tbb
  embree
  lz4::lz4
  PkgConfig::LuaJIT
  #
  Vulkan::Vulkan
  GPUOpen::VulkanMemoryAllocator
  glfw
  volk::volk_headers
  Vulkan::UtilityHeaders
  Freetype::Freetype
  ${SLANG_LIB}  
)

add_executable(keepsake_test)
target_sources(keepsake_test PRIVATE ${KS_SRC} ${KS_TEST_SRC} ${IMGUI_SRC})
target_compile_features(keepsake_test PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(keepsake_test PRIVATE /bigobj)
endif()
target_include_directories(keepsake_test PRIVATE ${COMMON_INCLUDE_DIRS})
if (MSVC)
  target_compile_options(keepsake_test PRIVATE /MP)
endif()
target_compile_definitions(keepsake_test PRIVATE ${COMMON_COMPILE_DEFS})
target_link_libraries(keepsake_test PRIVATE ${CPP_LIBS})
set_target_properties(keepsake_test PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # NOTE(grundlett): Really hacky. It appears there's no way to create plugin dll dependencies...
        # see https://gitlab.kitware.com/cmake/cmake/-/issues/22993
        add_custom_command(TARGET keepsake_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SLANG_DLL} $<TARGET_FILE_DIR:keepsake_test>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SLANG_GLSLANG_DLL} $<TARGET_FILE_DIR:keepsake_test>
        )
else()
        # TODO(grundlett) ...
        # find_file(SLANG_DLL libslang.so)
        # find_file(SLANG_GLSLANG_DLL libslang-glslang.so)
endif()

# if(WIN32)
#   # (Windows only) copy DLLs to the output directory.
#   foreach(dll ${SLANG_WIN32_DLLS})
#     add_custom_command(
#       TARGET gsr
#       POST_BUILD
#       COMMAND cmake -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:gsr>") 
#   endforeach()
# endif()