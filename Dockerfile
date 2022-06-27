FROM --platform=amd64 conda/miniconda3:latest

COPY environment.locked.yml .
RUN conda install -n base mamba -c conda-forge
RUN mamba env create --file environment.locked.yml
WORKDIR /app
RUN echo "source activate icenet" > ~/.bashrc
SHELL [ "/bin/bash", "-c"]
