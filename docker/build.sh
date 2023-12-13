#!/bin/bash
cp ../requirements.txt .
docker login git.ordbogen.com:5050
docker build -t git.ordbogen.com:5050/odin/odinllm/odinllm .

