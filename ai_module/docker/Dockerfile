FROM zhangjicmu/ubuntu24_ros:ai_module

# Base image runs as non-root user `docker` by default (confirmed via the
# `docker@...` shell prompt when exec'd into the running container) --
# apt-get needs root, so switch explicitly for the install steps below,
# then switch back so file ownership matches what the rest of this
# Dockerfile (and the original, unmodified version) expected.
USER root

# --- Python dependencies for vln_ai_module_dino. The base image's `pip` isn't
# guaranteed to be on PATH (confirmed missing as of 2026-08-07 -- only
# python3 itself is there), so install python3-pip explicitly first rather
# than assuming it exists. ---
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip curl zstd \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /home/docker/ai_module/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /home/docker/ai_module/requirements.txt

# --- Local LLM (Ollama + llama3.2:3b) for question_parser.py -- baked into
# the image at build time so the container is fully self-contained at grading
# time: no API key, no internet egress needed once the image is built. This
# does make the image larger (~2GB more) and the build slower (pulls the
# model once here) -- that's the trade for not depending on a secret in a
# publicly-pushed Docker Hub image.
#
# OLLAMA_MODELS is pinned to a fixed, world-writable path rather than left
# at Ollama's default ($HOME/.ollama/models). Without this, the model
# pulled here as root would land under /root/.ollama and be invisible to
# the `docker` user's `ollama serve` at container runtime (different
# $HOME) -- silently forcing a re-download over the network at grading
# time, which defeats the entire point of baking it in. ---
ENV OLLAMA_MODELS=/usr/share/ollama_models
RUN mkdir -p /usr/share/ollama_models && chmod -R 777 /usr/share/ollama_models
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN /bin/bash -c "\
    (ollama serve &) && \
    sleep 5 && \
    ollama pull llama3.2:1b"

# --- Grounding DINO (open-vocab object detector) weights, baked in for the
# same reason as the Ollama model above -- confirmed live during testing
# that a transient Hugging Face Hub 500 error crashed the node on startup
# (it was downloading weights fresh every single launch), which is exactly
# the fragility this avoids. Same HF_HOME pinning pattern as OLLAMA_MODELS
# above: baking as root here without pinning would put the cache under
# /root/.cache/huggingface, invisible to the `docker` user's runtime
# process (different $HOME) -- silently forcing a re-download at grading
# time and defeating the point of baking it in at all. ---
ENV HF_HOME=/usr/share/hf_cache
RUN mkdir -p /usr/share/hf_cache && chmod -R 777 /usr/share/hf_cache
RUN python3 -c "\
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; \
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')"

USER docker

# --- Copy source: keep dummy_vlm as a reference/fallback, and
# vln_ai_module_dino as the actual submitted model. ---
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY src/vln_ai_module_dino /home/docker/ai_module/src/vln_ai_module_dino

# Build both packages against the already-built workspace dependencies.
# NOTE: the base image ships a pre-built install/ (from the vendor's own
# dummy_vlm-only build). Building on top of that with --packages-select does
# NOT regenerate the top-level aggregating install/setup.bash to include the
# new package -- AMENT_PREFIX_PATH ends up missing it even though it
# installed correctly (confirmed via a live container: valid ament_index
# marker present, but `ros2 pkg list` still couldn't find it). Wiping
# install/build/log first forces a clean, consistent regeneration covering
# both packages in one pass.
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    rm -rf build install log && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm vln_ai_module_dino"

# Ensure the updated install is sourced in every new interactive shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc

# NOTE 2026-08-07: colcon's dynamic environment discovery
# (_local_setup_util_sh.py) produces ZERO output in this base image for
# EITHER package -- confirmed by running it directly in a live container and
# getting an empty result even for the vendor's own pre-built dummy_vlm.
# dummy_vlm "working" was never due to our .bashrc sourcing at all; it comes
# from a static AMENT_PREFIX_PATH the vendor baked into this base image
# directly. Since that discovery mechanism is broken/misconfigured here for
# every package, and an automated grading harness likely won't even use an
# interactive login shell (so .bashrc wouldn't be sourced regardless), set
# what vln_ai_module_dino needs explicitly and permanently via ENV rather
# than depending on colcon's aggregator.
ENV AMENT_PREFIX_PATH=/home/docker/ai_module/install/vln_ai_module_dino:/home/docker/ai_module/install/dummy_vlm:/opt/ros/jazzy
ENV PYTHONPATH=/home/docker/ai_module/install/vln_ai_module_dino/lib/python3.12/site-packages:${PYTHONPATH}
ENV PATH=/home/docker/ai_module/install/vln_ai_module_dino/lib/vln_ai_module_dino:${PATH}
