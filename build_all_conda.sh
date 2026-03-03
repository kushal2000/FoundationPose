PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Get pybind11 cmake dir from the conda environment
PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. -Dpybind11_DIR=${PYBIND11_DIR} && \
make -j$(nproc)

cd ${PROJ_ROOT}
