FROM ubuntu:latest

RUN apt-get update -q && apt-get install -yqq \
    apt-utils \
    git \
    vim \
    nano \
    ssh \
    gcc \
    make \
    build-essential \
    libkrb5-dev \
    sudo 
	
RUN apt-get update && apt-get install -y python python-dev python-pip \
    libxft-dev libfreetype6 libfreetype6-dev
RUN pip install 'matplotlib==1.4.3'
	
RUN apt-get install -y python python-dev python-distribute python-pip
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install boto3
RUN pip install -U boto
RUN pip install bokeh
RUN pip install jupyter
Run pip install sklearn
Run pip install pybrain
Run pip install seaborn
Run pip install scikit-learn
Run pip install Pillow
Run pip install fabulous
Run pip install urllib
Run pip install matplotlib
Run pip install bs4
Run pip install requests

ADD run.sh run.sh
ADD Makefile Makefile
ADD DataDownloading.py DataDownloading.py
ADD DataPreProcessing.py DataPreProcessing.py
ADD Prediction.py Prediction.py
ADD Prediction_whatIfAnalysis.py Prediction_whatIfAnalysis.py
ADD Classification_Part1.py Classification_Part1.py
ADD Classification_Part2.py Classification_Part2.py


#ENTRYPOINT ["bash","run.sh"]