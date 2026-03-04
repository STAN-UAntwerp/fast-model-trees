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

# Find Python (Development.Module is sufficient for extension modules;
# Development.Embed is not available in manylinux containers)
# NumPy is required by carma's headers
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

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
        # carma is header-only: use FetchContent_Populate to download without
        # calling add_subdirectory(), which would process carma's CMakeLists.txt.
        # carma's CMakeLists.txt uses cmake_minimum_required(VERSION < 3.5),
        # which CMake 4.x no longer supports.
        FetchContent_GetProperties(carma)
        if(NOT carma_POPULATED)
            FetchContent_Populate(carma)
        endif()
        set(CARMA_INCLUDE_DIR ${carma_SOURCE_DIR}/include)
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
target_include_directories(cpilot PRIVATE
    ${ARMADILLO_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}   # required by carma's headers
)

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