FROM --platform=amd64 conda/miniconda3:latest

COPY environment.locked.yml .
RUN conda install -n base mamba -c conda-forge
RUN mamba env create --file environment.locked.yml
WORKDIR /app
RUN echo "source activate icenet" > ~/.bashrc

ARG HDF5_USE_FILE_LOCKING=FALSE

SHELL [ "/bin/bash", "-c"]
