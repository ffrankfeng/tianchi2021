FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# prepare your environment here
RUN pip install imageio==2.9.0
RUN pip install matplotlib==3.0.3
RUN pip install pandas==1.1.3
RUN pip install Pillow==5.4.1
RUN pip install progressbar==2.5
RUN pip install scipy==1.1.0
RUN pip install timm==0.4.5
RUN pip install lpips==0.1.3
# RUN pip install ...

COPY code /workspace/code
WORKDIR /workspace/code
