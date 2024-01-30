# Docker builds an image based on the instruction in the dockerfile and the context defined by a path argument, i.e. <docker build -f path_to_dockerfile>.
# <docker build .> builds an image using all files and subdirectories in current working directory (which should only contain necessary files).
# To exclude particular files you can add a .dockerignore file to the context directory.
# You can also specify a repository and a tag, i.e. <docker build -t my_repo/my_image:tag -f path_to_dockerfile>.

# The dockerfile must start with a from statement that specifies a parent image.
FROM --platform=amd64 conda/miniconda3:latest   

# Set working directory inside image
WORKDIR /src

# Copy requirements.txt into /src directory
COPY environment.locked.yml /src

RUN conda install -n base mamba -c conda-forge

RUN mamba env create --file environment.locked.yml

RUN echo "source activate icenet" > ~/.bashrc

ARG HDF5_USE_FILE_LOCKING=FALSE

SHELL [ "/bin/bash", "-c"]