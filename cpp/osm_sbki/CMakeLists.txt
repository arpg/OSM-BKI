cmake_minimum_required(VERSION 3.16)

project(osm_sbki LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OSM_SBKI_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(OSM_SBKI_BUILD_PCL_EXAMPLES "Build PCL-based example programs" OFF)

set(OSM_SBKI_CORE_SOURCES
    src/continuous_bki.cpp
    src/file_io.cpp
    src/dataset_utils.cpp
)

add_library(osm_sbki_core STATIC ${OSM_SBKI_CORE_SOURCES})
target_include_directories(osm_sbki_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# OpenMP is optional; link it when available.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(osm_sbki_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(OSM_SBKI_ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(osm_sbki_core PRIVATE /W4)
    else()
        target_compile_options(osm_sbki_core PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(OSM_SBKI_BUILD_PCL_EXAMPLES)
    find_package(MPI REQUIRED COMPONENTS C)
    find_package(PCL QUIET COMPONENTS common io visualization)
    if(PCL_FOUND)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(LIBOSMIUM QUIET libosmium)
        endif()

        if(NOT LIBOSMIUM_FOUND)
            if(EXISTS "/usr/include/osmium/io/any_input.hpp")
                set(LIBOSMIUM_INCLUDE_DIRS "/usr/include")
                set(LIBOSMIUM_FOUND TRUE)
            elseif(EXISTS "/usr/local/include/osmium/io/any_input.hpp")
                set(LIBOSMIUM_INCLUDE_DIRS "/usr/local/include")
                set(LIBOSMIUM_FOUND TRUE)
            endif()
        endif()

        add_executable(visualize_map examples/visualize_map.cpp)
        target_include_directories(visualize_map PRIVATE ${PCL_INCLUDE_DIRS})
        target_link_libraries(visualize_map PRIVATE osm_sbki_core ${PCL_LIBRARIES})
        target_compile_options(visualize_map PRIVATE ${PCL_DEFINITIONS})

        if(LIBOSMIUM_FOUND)
            add_executable(visualize_osm examples/visualize_osm.cpp src/osm_parser.cpp)
            target_include_directories(visualize_osm PRIVATE ${PCL_INCLUDE_DIRS} ${LIBOSMIUM_INCLUDE_DIRS})
            target_link_libraries(visualize_osm PRIVATE osm_sbki_core ${PCL_LIBRARIES} expat bz2 z)
            target_compile_options(visualize_osm PRIVATE ${PCL_DEFINITIONS})
        else()
            message(STATUS "libosmium headers not found. Skipping visualize_osm target.")
        endif()
    else()
        message(STATUS "PCL not found. Skipping PCL example targets.")
    endif()
endif()
