#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025 Calvin Vette

# TODO
# There has to be a cleaner way of doing this...

mkdir -p ../lib/hopper


docker build . \
    -f Dockerfile.flash-attention-cu126-cp312-hopper.amd64 \
    -t nge/builder.flash-attention \
    && \
docker run \
        --rm \
        -it \
        -v ../lib:/out \
        --name booger nge/builder.flash-attention \
        cp /workspace/flash-attention/dist/flash_attn-2.8.3-cp312-cp312-linux_$(arch).whl /out/hopper
