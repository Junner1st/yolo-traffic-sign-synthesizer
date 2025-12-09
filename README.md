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
python main.py
```

## Data
```sh
wget https://github.com/exodustw/Taiwan-Traffic-Sign-Recognition-Benchmark/releases/download/v1/ttsrb_v1.zip
unzip ttsrb_v1.zip -d data
mv data/ttsrb data/signs
```