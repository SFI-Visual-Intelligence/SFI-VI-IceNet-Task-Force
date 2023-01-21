## Must specify architecture for successful execution on cluster
FROM --platform=amd64 conda/miniconda3:latest

COPY environment.locked.yml .
## Install mamba (conda alternative)
RUN conda install -n base mamba -c conda-forge

## Setup environment with all packages
RUN mamba env create --file environment.locked.yml

## Set working directory in container
WORKDIR /app

## Activate environment automatically when opening bash shell
RUN echo "source activate icenet" > ~/.bashrc

## Env variable added to open cmip6 data.
ARG HDF5_USE_FILE_LOCKING=FALSE

## Open bash shell
SHELL ["/bin/bash", "-c"]
