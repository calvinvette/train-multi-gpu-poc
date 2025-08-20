#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025 Calvin Vette

# TODO
# There has to be a cleaner way of doing this...

mkdir -p ../out


docker build . \
    -f Dockerfile.flash-attention-cu129-cp312.amd64 \
    -t nge/builder.flash-attention \
    && \
docker run \
        --rm \
        -it \
        -v ../out:/out \
        --name booger nge/builder.flash-attention \
        cp /workspace/flash-attention/dist/flash_attn-2.8.2-cp312-cp312-linux_x86_64.whl /out/
