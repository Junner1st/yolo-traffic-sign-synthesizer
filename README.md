# Yolo Traffic Sign Synthesizer

## Build Environment
```
conda env create -f environment.yml
```

Get into the environment:
```
conda activate synthesizer_311
```

## Run
```
python main.py
```

## Data
```
wget https://github.com/exodustw/Taiwan-Traffic-Sign-Recognition-Benchmark/releases/download/v1/ttsrb_v1.zip
unzip ttsrb_v1.zip -d data
mv data/ttsrb data/signs
```