# some useful invocations

fine-tuning mistral for Danish intent classification on 8 GPUs
```
NCCL_SOCKET_IFNAME=bond0.100 NCCL_P2P_LEVEL=NVL ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_ip=10.10.0.21 --main_process_port=29500 --config_file config/accelerate/multi_gpu.yaml --num_processes=8 --num_machines=2 --machine_rank=0 -m odinllm train ../abc-instruct/config/train-sft.yaml models/mistral-7b models/mistral-abc
```

fine-tuning mixtral for Danish Q&A on 4 GPUs:
```
NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=LOC ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=29500 --config_file config/accelerate/multi_gpu.yaml --num_processes=4 -m odinllm train config/default.yaml config/train/sft-qa.yaml models/mixtral models/mixtral-qa
```
