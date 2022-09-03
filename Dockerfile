FROM nvcr.io/nvidia/tritonserver:22.05-py3
RUN pip install --upgrade pip
RUN mkdir -p /cv_root/models
COPY ./requirements.txt /cv_root
COPY ./synergy.csv /cv_root
COPY models /cv_root/models
RUN pip install -r /cv_root/requirements.txt
CMD ["tritonserver", "--model-repository=/cv_root/models", "--allow-grpc=false", "--allow-http=true", "--http-port=8080", "--allow-metrics=false", "--allow-gpu-metrics=false"]
