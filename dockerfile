FROM fogify-jupiter AS builder


FROM andreper/jupyterlab:3.0.0-spark-3.0.0

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev && \
    pip3 install --upgrade pip && \
    pip3 install pyspark==3.0.0 && \
    pip3 install edgerun-ether==0.3.3

RUN pip3 install FogifySDK

COPY --from=builder /opt/conda/lib/python3.9/site-packages/FogifySDK /usr/local/lib/python3.9/dist-packages/FogifySDK


RUN pip3 install PyYAML
RUN pip3 install tqdm
RUN pip3 install prometheus-pandas==0.3.1

WORKDIR /home/jovyan/work
# CMD ["start-notebook.sh"]