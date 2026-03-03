fp()
{
    eval "$(/juno/u/kedia/miniconda3/bin/conda shell.bash hook)"
    eval "$(mamba shell hook --shell bash)"
    export CUDA_HOME=/usr/local/cuda-11.8
    export CUDA_PATH=$CUDA_HOME
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
    export PATH=${CUDA_HOME}/bin:$PATH
    mamba deactivate
    mamba deactivate
    mamba activate fp

    alias killros='ps aux | grep ros | grep tylerlum | awk '\''{print $2}'\'' | xargs kill -9'
    export UV_CACHE_DIR=/juno/u/kedia/.cache
    export ROS_MASTER_URI=http://bohg-ws-2.stanford.edu:11311
    export ROS_HOSTNAME=bohg-ws-16.stanford.edu
    export ROS_IP=bohg-ws-16.stanford.edu
    # kedia
    cd /juno/u/kedia/FoundationPose/
    export CUDA_HOME=/usr/local/cuda-11.8
    export CUDA_PATH=$CUDA_HOME
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
    export PATH=${CUDA_HOME}/bin:$PATH
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
    export ROS_HOME=/juno/u/kedia/.ros
    export ROS_LOG_DIR=/juno/u/kedia/.ros
    export HOME=/juno/u/kedia/
    export ROS_HOME="/scr2/kedia"
    export ROS_LOG_DIR="/scr2/kedia"
    export HOME="/scr2/kedia"
    export ROS_HOME="/home/kedia"
    export ROS_LOG_DIR="/home/kedia"
    export HOME="/home/kedia"
    export TORCH_EXTENSIONS_DIR="/home/kedia"
}
