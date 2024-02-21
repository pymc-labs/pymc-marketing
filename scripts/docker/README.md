# Docker Image

To build the docker image, run the following command:

```bash
docker build  -f scripts/docker/Dockerfile -t docker-pymc-marketing .
```

To run the docker image, run the following command:

```bash
docker run -it -p 8888:8888 -v $(pwd):/home/jovyan/work docker-pymc-marketing
```
