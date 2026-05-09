# WaveMixerNet

This is the official PyTorch implementation of WaveMixerNet: A Dual-Stream Wavelet-based MLP-Mixer and CNN Architecture for Long-Term Time Series Forecasting.

🚩 **News** (2026.05) To facilitate the peer review process, we have made the complete code publicly available.

🚩 **News** (2025.11) ~~The complete code will be uploaded after the paper is accepted.~~

## Environment

- Ubuntu 22.04.3 LTS
- Python 3.10.14
- Pytorch 2.0.1
- CUDA 11.8
- GPU NVIDIA GeForce RTX 4090 24G

## Getting Started

1. Install requirements.

    ```
    pip install -r requirements.txt
    ```

2. Download data. You can download all the datasets from [Autoformer](https://github.com/thuml/Autoformer) or [TimeMixer](https://github.com/kwuking/TimeMixer). Create a separate folder `./data` and put all the csv files in the directory.

    ```
    data
    ├── electricity
    │   └── electricity.csv
    ├── ETT
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    │   └── ETTm2.csv
    │
    ├── traffic
    │   └── traffic.csv
    └── weather
        └── weather.csv
    ```

3. Training. All the scripts are in the directory `./scripts/long_term_forecast`. For example, if you want to get the multivariate forecasting results for ETTh1 dataset, just run the following command, and you can open `./result.txt` to see the results once the training is done:

    ```
    bash ./scripts/long_term_forecast/etth1.sh
    ```

    You can also run all the datasets sequentially by:

    ```
    bash ./scripts/run_all.sh
    ```

You can adjust the hyperparameters based on your needs, such as patch length, look-back window, and prediction length.

## Acknowledgement

We sincerely appreciate the following GitHub repositories for their valuable codebases and datasets:

- https://github.com/cure-lab/LTSF-Linear
- https://github.com/yuqinie98/PatchTST
- https://github.com/decisionintelligence/DUET
- https://github.com/kwuking/TimeMixer
- https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer
- https://github.com/Hank0626/PDF
- https://github.com/luodhhh/ModernTCN

## Contact

If you have any questions or concerns, please contact us at bwang2763@gmail.com or wangbo2024@stumail.hbu.edu.cn, or submit an issue.
