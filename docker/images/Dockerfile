# 1) Base image
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV CUDA_HOME=/usr/local/cuda-11.1/

# Tell Debian to not ask any questions, and pick our timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

# 2) Remove the broken NVIDIA repos (they lack their key in this image)
RUN rm -f \
      /etc/apt/sources.list.d/*cuda*.list \
      /etc/apt/sources.list.d/*machine-learning*.list \
      /etc/apt/sources.list.d/*nvidia*.list

# 3) Install tzdata non-interactively, plus sudo, wget, unzip, python3-opencv
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
      sudo \
      wget \
      unzip \
      python3-opencv \
 # configure the timezone without a prompt:
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# 4) Create non-root "admin" user with password-less sudo
# ARG USERNAME=admin
# ARG USER_ID=1000
# ARG GROUP_ID=1000

# RUN groupadd -g ${GROUP_ID} ${USERNAME} \
#  && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} \
#  && mkdir -p /etc/sudoers.d \
#  && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
#       | tee /etc/sudoers.d/${USERNAME} \
#  && chmod 440 /etc/sudoers.d/${USERNAME} \
#  && usermod -aG sudo,video ${USERNAME}

# 5) Python packages via pip
RUN pip install --no-cache-dir \
      mmcv-full==1.4.4 \
      mmsegmentation==0.22.1 \
      timm \
      tqdm \
      thop \
      pandas \
      tensorboard \
      ipdb \
      h5py \
      ipython \
      Pillow==9.5.0 \
 && pip install -U numpy

# 6) Copy your code and build any extensions
WORKDIR /completion_former
COPY . .
# RUN cd src/model/deformconv/ && CUDA_HOME=/usr/local/cuda-11.1/ python setup.py build install

# Install NVIDIA Apex
ARG COMMIT=4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
RUN wget -q https://github.com/NVIDIA/apex/archive/${COMMIT}.zip \
 && unzip -q ${COMMIT}.zip \
 && rm ${COMMIT}.zip \
 && cd apex-${COMMIT} \
 && pip install -v --no-cache-dir \
       --global-option="--cpp_ext" \
       --global-option="--cuda_ext" ./

# 7) Add entrypoint script
COPY /docker/entrypoints/build_dcn_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
# USER ${USERNAME}

# Use entrypoint
ENTRYPOINT ["/entrypoint.sh"]
# CMD ["/bin/bash"]
