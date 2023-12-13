#!/bin/bash
docker login git.ordbogen.com:5050
docker run --gpus all -it git.ordbogen.com:5050/odin/odinllm/odinllm

