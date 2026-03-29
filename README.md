# Yolo Traffic Sign Synthesizer

## Load submodule
After cloning:
```sh
git submodule update --init --recursive
```

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

### 1. Frame Extract
```sh
cd src
python frame_extract.py
```

### 2. Synthesized Data
```sh
cd src
python main.py
```

### 3. Train Yolo
```sh
cd src
python main.py
```

### 4. Recognition Sign For a Video
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
