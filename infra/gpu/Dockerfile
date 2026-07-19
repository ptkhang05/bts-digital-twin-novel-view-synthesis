FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e

ARG HOST_UID
ARG HOST_GID

ENV HOME=/tmp \
    TORCH_HOME=/opt/bts-nvs/torch-cache

RUN test -n "$HOST_UID" && test -n "$HOST_GID" && \
    if ! getent group "$HOST_GID" >/dev/null; then \
        groupadd --gid "$HOST_GID" "btsnvs-${HOST_GID}"; \
    fi && \
    if ! getent passwd "$HOST_UID" >/dev/null; then \
        useradd --uid "$HOST_UID" --gid "$HOST_GID" \
        --home-dir /tmp --no-create-home --shell /usr/sbin/nologin \
        "btsnvs-${HOST_UID}"; \
    fi && \
    getent passwd "$HOST_UID" >/dev/null

# Metric-only dependencies. LPIPS weights are never inputs to training/rendering.
COPY infra/gpu/constraints-metrics.txt /tmp/constraints-metrics.txt
COPY infra/gpu/check_pip_environment.py /tmp/check-pip-environment.py
RUN mkdir -p "$TORCH_HOME" && \
    python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --constraint /tmp/constraints-metrics.txt \
    lpips==0.1.4 \
    scikit-image==0.25.2 && \
    python /tmp/check-pip-environment.py && \
    python -c "import lpips; lpips.LPIPS(net='alex')" && \
    chmod -R a+rX /opt/bts-nvs/torch-cache && \
    rm /tmp/constraints-metrics.txt /tmp/check-pip-environment.py

ENV PYTHONPATH=/workspace/src
