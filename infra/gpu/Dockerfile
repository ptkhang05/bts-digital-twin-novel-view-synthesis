FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e

# Metric-only dependencies. LPIPS weights are never inputs to training/rendering.
COPY infra/gpu/constraints-metrics.txt /tmp/constraints-metrics.txt
RUN python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --constraint /tmp/constraints-metrics.txt \
    lpips==0.1.4 \
    scikit-image==0.25.2 && \
    python -m pip check && \
    python -c "import lpips; lpips.LPIPS(net='alex')" && \
    rm /tmp/constraints-metrics.txt

ENV PYTHONPATH=/workspace/src
