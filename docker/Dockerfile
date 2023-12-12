FROM continuumio/miniconda3
RUN conda create -y -n env python=3.10
ADD requirements.txt /
RUN conda init bash
RUN bash -c "source /root/.bashrc && conda activate env && pip install -r /requirements.txt"
RUN bash -c "source /root/.bashrc && conda activate env && conda install -y -c nvidia cuda-nvcc"
RUN bash -c "source /root/.bashrc && conda activate env && python -m pip install flash-attn --no-build-isolation"
CMD [ "/bin/bash" ]

