# Yolo Traffic Sign Synthesizer

## Build Environment
```sh
conda env create -f environment.yml
```

To get into the environment:
```sh
conda activate synthesizer_311
```

To export the environment:
```sh
conda env export | sed '$d' > environment.yml
```

## Run
```sh
cd src
python main.py
```

Run recognition sign:
```sh
cd src
python recognition.py
```
Video output will store at `data/recognized`


## Download Sign Data
```sh
wget https://github.com/exodustw/Taiwan-Traffic-Sign-Recognition-Benchmark/releases/download/v1/ttsrb_v1.zip
unzip ttsrb_v1.zip -d data
mv data/ttsrb data/signs
```
