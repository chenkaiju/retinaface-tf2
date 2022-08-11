FROM tensorflow/tensorflow:2.8.0-gpu

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    nvidia-cuda-toolkit\
 && rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /workspace
WORKDIR /workspace

# Install python packages
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install nvidia-pyindex
RUN pip install nvidia-tensorrt


#### Add non-root user
ARG USERNAME=kasper
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USER_UID:$USER_GID /workspace

USER $USERNAME

# All users can use /home/user as their home directory
ENV HOME=/home/$USERNAME
RUN chmod 777 /home/$USERNAME
    
# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
# # Set up the Conda environment
# ENV CONDA_AUTO_UPDATE_CONDA=false \
#     PATH=/home/$USERNAME/miniconda/bin:$PATH
# COPY environment.yml /app/environment.yml
# RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
#  && chmod +x ~/miniconda.sh \
#  && ~/miniconda.sh -b -p ~/miniconda \
#  && rm ~/miniconda.sh \
#  && conda env update -n base -f /app/environment.yml \
#  && rm /app/environment.yml \
#  && conda clean -ya





# [Optional] Set the default user. Omit if you want to keep the default as root.

