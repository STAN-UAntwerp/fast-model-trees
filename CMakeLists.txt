cmake_minimum_required(VERSION 3.12)

# Auto-detect vcpkg toolchain on Windows if VCPKG_INSTALLATION_ROOT is set
# (automatically set on GitHub Actions Windows runners and many local vcpkg installs)
if(WIN32 AND DEFINED ENV{VCPKG_INSTALLATION_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file" FORCE)
    message(STATUS "Auto-detected vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

project(pilot_wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find linear algebra libraries
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Find carma: try installed package, then common header paths, then download automatically
find_package(carma CONFIG QUIET)
if(NOT carma_FOUND)
    message(STATUS "carma not found via find_package, trying to locate manually...")
    find_path(CARMA_INCLUDE_DIR
        NAMES carma
        PATHS
            "/usr/local/include"
            "/usr/include"
            "$ENV{HOME}/.local/include"
        PATH_SUFFIXES include
    )
    if(CARMA_INCLUDE_DIR)
        message(STATUS "Found carma headers at: ${CARMA_INCLUDE_DIR}")
        set(carma_FOUND TRUE)
    else()
        message(STATUS "carma not found locally, fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            carma
            URL https://github.com/RUrlus/carma/archive/refs/heads/stable.tar.gz
        )
        FetchContent_MakeAvailable(carma)
        set(carma_FOUND TRUE)
    endif()
endif()

# Create the Python extension module
pybind11_add_module(cpilot pilot_wrapper.cpp tree.cpp)

# Link libraries
target_link_libraries(cpilot PRIVATE
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Include directories
target_include_directories(cpilot PRIVATE ${ARMADILLO_INCLUDE_DIRS})

# Link carma if found via find_package
if(TARGET carma::carma)
    target_link_libraries(cpilot PRIVATE carma::carma)
elseif(CARMA_INCLUDE_DIR)
    target_include_directories(cpilot PRIVATE ${CARMA_INCLUDE_DIR})
endif()

# Install the module
# For scikit-build-core, we need to install to ${SKBUILD_PLATLIB_DIR}
if(SKBUILD)
    install(TARGETS cpilot LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/pilot)
else()
    install(TARGETS cpilot LIBRARY DESTINATION pilot)
endif()