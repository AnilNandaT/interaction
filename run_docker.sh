#!/bin/bash
docker run --rm \
     --shm-size=5g --ulimit memlock=-1 --ulimit stack=67108864 \
     --name drug-interaction \
     -p 8080:8080 \
     drug-interaction
