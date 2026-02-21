cmake_minimum_required(VERSION 3.16)

project(osm_sbki LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OSM_SBKI_ENABLE_WARNINGS "Enable compiler warnings" ON)
# Use set(... CACHE BOOL ...) so that -DOSM_SBKI_BUILD_PCL_EXAMPLES=ON on the command line is applied even when reconfiguring (option() would keep the first cached value).
set(OSM_SBKI_BUILD_PCL_EXAMPLES OFF CACHE BOOL "Build PCL-based example program (visualize_map_osm)")
set(OSMIUM_INCLUDE_DIR "" CACHE PATH "Path to libosmium include directory (e.g. /path/to/osmium/include; required for visualize_map_osm)")

message(STATUS "OSM_SBKI_BUILD_PCL_EXAMPLES=${OSM_SBKI_BUILD_PCL_EXAMPLES}")

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
    message(STATUS "PCL example requested; looking for MPI, PCL, and libosmium...")
    find_package(MPI REQUIRED COMPONENTS C)
    find_package(PCL QUIET COMPONENTS common io visualization)
    message(STATUS "PCL found: ${PCL_FOUND}")

    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(LIBOSMIUM QUIET libosmium)
    endif()
    if(NOT LIBOSMIUM_FOUND)
        if(OSMIUM_INCLUDE_DIR AND EXISTS "${OSMIUM_INCLUDE_DIR}/osmium/io/any_input.hpp")
            set(LIBOSMIUM_INCLUDE_DIRS "${OSMIUM_INCLUDE_DIR}")
            set(LIBOSMIUM_FOUND TRUE)
        elseif(EXISTS "/usr/include/osmium/io/any_input.hpp")
            set(LIBOSMIUM_INCLUDE_DIRS "/usr/include")
            set(LIBOSMIUM_FOUND TRUE)
        elseif(EXISTS "/usr/local/include/osmium/io/any_input.hpp")
            set(LIBOSMIUM_INCLUDE_DIRS "/usr/local/include")
            set(LIBOSMIUM_FOUND TRUE)
        endif()
    endif()
    message(STATUS "libosmium found: ${LIBOSMIUM_FOUND}")

    if(PCL_FOUND AND LIBOSMIUM_FOUND)
        message(STATUS "Adding example: visualize_map_osm")
        add_executable(visualize_map_osm examples/visualize_map_osm.cpp src/osm_parser.cpp)
        target_include_directories(visualize_map_osm PRIVATE ${PCL_INCLUDE_DIRS} ${LIBOSMIUM_INCLUDE_DIRS})
        target_link_libraries(visualize_map_osm PRIVATE osm_sbki_core ${PCL_LIBRARIES} expat bz2 z)
        target_compile_options(visualize_map_osm PRIVATE ${PCL_DEFINITIONS})
    else()
        if(NOT PCL_FOUND)
            message(STATUS "PCL not found. Skipping example target visualize_map_osm.")
        endif()
        if(NOT LIBOSMIUM_FOUND)
            message(STATUS "libosmium not found. Set OSMIUM_INCLUDE_DIR to enable visualize_map_osm.")
        endif()
    endif()
endif()
