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
    export UV_CACHE_DIR=/home/tylerlum/.cache
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
    export ROS_HOME=/home/tylerlum/.ros
    export ROS_LOG_DIR=/home/tylerlum/.ros
    export HOME=/home/tylerlum/
    export ROS_HOME="/home/tylerlum"
    export ROS_LOG_DIR="/home/tylerlum"
    export HOME="/home/tylerlum"
    export ROS_HOME="/home/tylerlum"
    export ROS_LOG_DIR="/home/tylerlum"
    export HOME="/home/tylerlum"
    export TORCH_EXTENSIONS_DIR="/home/tylerlum"
}
