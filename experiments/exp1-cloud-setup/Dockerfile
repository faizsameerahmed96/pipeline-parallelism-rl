FROM python:3.11-slim

RUN apt-get update && apt-get install -y swig build-essential

RUN pip install \
    "tensorboard>=2.10.0" \
    "wandb>=0.13.11" \
    "gym==0.26.2" \
    "gymnasium[box2d]==1.2.0" \
    "glfw>=2.6" \
    "mujoco==3.3.6" \
    "torch==2.4.1" \
    "moviepy>=1.0.3" \
    "pygame>=2.1" \
    "huggingface-hub>=0.11.1" \
    "rich<12.0" \
    "tenacity>=8.2.2" \
    "tyro>=0.5.10" \
    "swig>=4.3.1.post0" \
    "torchviz>=0.0.3" \
    "torchinfo>=1.8.0" \
    "psutil>=5.9.0"

WORKDIR /workspace
COPY . /workspace

ENTRYPOINT ["python"]