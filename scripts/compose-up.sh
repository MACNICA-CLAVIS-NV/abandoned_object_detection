#!/usr/bin/env bash

set -eu

source scripts/docker_base.sh

cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list .
cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc .

export BASE_IMAGE=${BASE_IMAGE}
export L4T_VERSION=${L4T_VERSION}

sudo -E ~/.local/bin/docker-compose up
