This is an implementation for the PIecewise Linear Organic Tree (PILOT), a linear model tree algorithm proposed in the paper Raymaekers, J., Rousseeuw, P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by PILOT. Machine Learning, 1-50. https://doi.org/10.1007/s10994-024-06590-3.

This repository also includes the implementation for RaFFLE, a random forest of PILOT trees:
Raymaekers, J., Rousseeuw, P. J., Servotte, T., Verdonck, T., & Yao, R. (2025). A Powerful Random Forest Featuring Linear Extensions (RaFFLE). _Under Review_

### Requirements:
Requirements can be installed by running
```
pip install -r requirements.txt
```
The RaFFLE implementation uses a c++ version of pilot for computational speed.
To build the c++ wrapper, follow these steps.

### Linux

1.  Make sure the necessary dependencies are installed
    ```
    sudo apt-get update
    sudo apt-get install cmake g++ libopenblas-dev liblapack-dev
    ```

2.  Install Armadillo
    ```
    wget http://sourceforge.net/projects/arma/files/armadillo-14.0.3.tar.xz
    tar -xvf armadillo-14.0.3.tar.xz
    cd armadillo-14.0.3/
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    ```

3.  Install pybind and add to cmake config
    ```
    pip install pybind11
    ```
    Locate the cmake config:
    ```
    pip show pybind11
    ```
    copy the installation path to `CMakeLists.txt`:
    ```
    set(pybind11_DIR <path>/pybind11/share/cmake/pybind11)
    ```

4.  Install carma
    Clone the repo: `git@github.com:RUrlus/carma.git`
    Build the package:
    ```
    cd carma
    mkdir build
    cd build
    cmake -DCARMA_INSTALL_LIB=ON ..
    cmake --build . --config Release --target install
    ```

5.  Build the wrapper
    Create a `build` directory in the root of the project, and `cd` into it.
    ```
    mkdir build
    cd build
    cmake ..
    make
    ```

### macOS

1.  **Prerequisites:**
    *   Make sure you have [Homebrew](https://brew.sh/) installed.
    *   Install Xcode Command Line Tools by running `xcode-select --install`.

2.  **Dependencies:**
    Install `cmake`, `openblas`, `lapack` and `armadillo` using Homebrew:
    ```
    brew install cmake openblas lapack armadillo
    ```

3.  **pybind11:**
    Install `pybind11` using pip:
    ```
    pip install pybind11
    ```
    Then, you need to tell cmake where to find the `pybind11` cmake files. Add the following line to your `CMakeLists.txt` *before* the `find_package(pybind11 REQUIRED)` line:
    ```cmake
    set(pybind11_DIR <path-to-your-venv>/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
    ```
    You can find the path to your virtual environment's `site-packages` by running `python -m site --user-site`.

4.  **carma:**
    Clone the `carma` repository and build it from source. Note that the last command might require `sudo`.
    ```
    git clone git@github.com:RUrlus/carma.git
    cd carma
    mkdir build
    cd build
    cmake -DCARMA_INSTALL_LIB=ON ..
    cmake --build . --config Release --target install
    ```

5.  **Build the wrapper:**
    Create a `build` directory in the root of the project, and `cd` into it.
    ```
    mkdir build
    cd build
    cmake ..
    make
    ```

### Troubleshooting

If you get an error like this on macOS:
```
Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS NumPy)
```
You might have to explicitly set the Python executable and library paths in your `CMakeLists.txt`. Add the following lines *before* the `find_package(Python3 ...)` line:

```cmake
set(Python3_EXECUTABLE <path-to-your-python-executable>)
set(Python3_LIBRARY <path-to-your-python-library>)
```

You can find the paths by running the following commands (make sure you are in your project's virtual environment):
*   For `Python3_EXECUTABLE`:
    ```
    which python
    ```
*   For `Python3_LIBRARY`: This path might vary. A good guess is to look for a `.dylib` file in the `lib` directory of your python installation. For example:
    ```
    /Users/user/.local/share/uv/python/cpython-3.10.18-macos-aarch64-none/lib/libpython3.10.dylib
    ```

Also, you might need to simplify the `find_package(Python3 ...)` call to:
```cmake
find_package(Python3 COMPONENTS Interpreter REQUIRED)
```


### Example
You can run an example for RaFFLE with the [raffle_example.py](raffle_example.py) script.

### RaFFLE benchmark
To run the same benchmark as described in the RaFFLE paper, you first need to download all the benchmark datasets using the [download_data.py](download_data.py) script.

```
python download_data.py
```

Next you can run benchmark by running the [benchmark.py](benchmark.py) script:
```
python benchmark.py
```

Results will be stored in the `Output` folder.

The plots from the paper are created with the [paperplots.py](paperplots.py) script. You can create all plots by running:

```
python paperplots.py --all
```

### Local Modifications

The `CMakeLists.txt` file might need to be modified locally to set the correct paths for your system's dependencies. To prevent these local changes from being committed to the repository, you can tell Git to assume that the file hasn't changed.

To do this, run the following command:

```
git update-index --assume-unchanged CMakeLists.txt
```

This will prevent your local changes from being tracked. If you later need to pull updates from the remote repository and want to re-apply your local changes, you can reverse this with:

```
git update-index --no-assume-unchanged CMakeLists.txt
```

And then, if you want to discard your local changes and get the version from the repository:

```
git checkout -- CMakeLists.txt
```
