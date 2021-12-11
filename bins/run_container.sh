#!/bin/bash

image_name="nv_report"
username="ISedunov"
container_name=${username}-${image_name}

docker stop "${container_name}"
docker rm "${container_name}"

docker run -it \
    --gpus all \
    --expose 22 -P \
    --shm-size 8G \
    --runtime=nvidia \
    -v $PWD/../../:/home/neural_vocoder \
    --detach \
    --name "${container_name}" \
    --entrypoint /bin/bash \
    ${image_name}