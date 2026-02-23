cmake_minimum_required(VERSION 3.16)

project(osm_bki LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(osm_bki_ENABLE_WARNINGS "Enable compiler warnings" ON)
# Use set(... CACHE BOOL ...) so that -Dosm_bki_BUILD_PCL_EXAMPLES=ON on the command line is applied even when reconfiguring (option() would keep the first cached value).
set(osm_bki_BUILD_PCL_EXAMPLES OFF CACHE BOOL "Build PCL-based example program (visualize_map_osm)")
set(OSMIUM_INCLUDE_DIR "" CACHE PATH "Path to libosmium include directory (e.g. /path/to/osmium/include; required for visualize_map_osm)")
set(Boost_INCLUDE_DIR_MANUAL "" CACHE PATH "Override: path to Boost include (directory that contains boost/; set if PCL fails to find boost/mpl/assert.hpp)")

message(STATUS "osm_bki_BUILD_PCL_EXAMPLES=${osm_bki_BUILD_PCL_EXAMPLES}")

set(osm_bki_CORE_SOURCES
    src/continuous_bki.cpp
    src/file_io.cpp
    src/dataset_utils.cpp
    src/osm_loader.cpp
)

add_library(osm_bki_core STATIC ${osm_bki_CORE_SOURCES})
target_include_directories(osm_bki_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Eigen3 for vectorized linear algebra
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(osm_bki_core PUBLIC Eigen3::Eigen)

# OpenMP is optional; link it when available.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(osm_bki_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(osm_bki_ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(osm_bki_core PRIVATE /W4)
    else()
        target_compile_options(osm_bki_core PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(osm_bki_BUILD_PCL_EXAMPLES)
    message(STATUS "PCL example requested; looking for Boost, MPI, PCL, and libosmium...")
    # PCL requires Boost (e.g. boost/mpl/assert.hpp). Find it before PCL.
    set(Boost_NO_BOOST_CMAKE ON)  # use FindBoost.cmake, not BoostConfig.cmake
    find_package(Boost 1.65 REQUIRED)
    if(NOT Boost_INCLUDE_DIRS)
        set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})
    endif()
    if(Boost_INCLUDE_DIR_MANUAL)
        set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR_MANUAL})
        message(STATUS "Using manual Boost include: ${Boost_INCLUDE_DIR_MANUAL}")
    endif()
    message(STATUS "Boost found: ${Boost_VERSION} at ${Boost_INCLUDE_DIRS}")
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

    if(PCL_FOUND)
        message(STATUS "Adding example: test_cbki (continuous mapping + PCL viz)")
        add_executable(test_cbki examples/test_cbki.cpp)
        # PCL headers #include <boost/...>; Boost must be in include path before PCL
        target_include_directories(test_cbki PRIVATE
            ${Boost_INCLUDE_DIRS}
            ${PCL_INCLUDE_DIRS}
        )
        target_link_libraries(test_cbki PRIVATE osm_bki_core ${PCL_LIBRARIES})
        if(TARGET Boost::headers)
            target_link_libraries(test_cbki PRIVATE Boost::headers)
        endif()
        target_compile_options(test_cbki PRIVATE ${PCL_DEFINITIONS})
    endif()

    if(PCL_FOUND AND LIBOSMIUM_FOUND)
        message(STATUS "Adding example: visualize_map_osm")
        add_executable(visualize_map_osm examples/visualize_map_osm.cpp src/osm_parser.cpp)
        target_include_directories(visualize_map_osm PRIVATE ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${LIBOSMIUM_INCLUDE_DIRS})
        target_link_libraries(visualize_map_osm PRIVATE osm_bki_core ${PCL_LIBRARIES} expat bz2 z)
        if(TARGET Boost::headers)
            target_link_libraries(visualize_map_osm PRIVATE Boost::headers)
        endif()
        target_compile_options(visualize_map_osm PRIVATE ${PCL_DEFINITIONS})
    else()
        if(NOT PCL_FOUND)
            message(STATUS "PCL not found. Skipping example targets.")
        endif()
        if(NOT LIBOSMIUM_FOUND)
            message(STATUS "libosmium not found. Set OSMIUM_INCLUDE_DIR to enable visualize_map_osm.")
        endif()
    endif()
endif()
