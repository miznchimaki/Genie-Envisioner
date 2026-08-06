#!/usr/bin/bash


source ${HOME}/.bashrc

script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

if [ -z $WORLD_SIZE ]; then
NGPU=`nvidia-smi --list-gpus | wc -l`
NGPU=6

# TODO: (mizuno) The following environmental variable is useful for torch2.6/2.7/2.8+cu126 + Python3.13
# export TORCHINDUCTOR_COMPILE_THREADS=1

cd ${HOME}/projects/Genie-Envisioner/
echo "Training on 1 Nodes, $NGPU GPUs"
# TODO: (mizuno) The following command is for torch2.6/2.7/2.8+cu126 + Python3.13
# lerobot310-python -m torch.distributed.run --nnodes=1 \
#     --nproc_per_node=$NGPU \
#     --node_rank=0 \
#     --no_python \
#     lerobot310-python \
#     $script_path \
#     --config_file $config_path
lerobot310-python -m torch.distributed.run --nnodes=1 \
    --nproc_per_node=$NGPU \
    --node_rank=0 \
    $script_path \
    --config_file $config_path  2>&1 | tee logs/debug.log
else
echo "Training on $WORLD_SIZE Nodes, 8 GPU per Node"
NGPU=`nvidia-smi --list-gpus | wc -l`
lerobot310-python -m torch.distributed.run --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NGPU \
    --node_rank=$RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $script_path \
    --config_file $config_path
fi
