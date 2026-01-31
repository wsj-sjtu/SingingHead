# SingingHead: A Large-scale 4D Dataset for Singing Head Animation
## [arXiv](https://arxiv.org/pdf/2312.04369.pdf) | [Project Page](https://wsj-sjtu.github.io/SingingHead/) | [Dataset](https://huggingface.co/datasets/Human-X/SingingHead)

<img src="assets/teaser.png" /> 

## TODO
- [x] Release the codes for calculating the metrics of two benchmarks.
- [ ] Release the scripts for visualizing the 3D facial motion.
- [x] Release the SingingHead dataset.

## SingingHead Dataset
### Download
The dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/Human-X/SingingHead).

If you are unable to download from Hugging Face, please first fill out the required information on [Hugging Face](https://huggingface.co/datasets/Human-X/SingingHead) to obtain authorization, and then contact us [(wusijing@sjtu.edu.cn)](wusijing@sjtu.edu.cn) using the same email address to get the download link of Baidu (百度网盘).

Please note that by requesting the dataset, you confirm that you have read, understood, and agree to be bound by the terms of the agreement.

**Agreement**

1. The SingingHead dataset is available for **non-commercial** research purposes only.

2. You agree **not to** reproduce, modified, duplicate, copy, sell, trade, resell or exploit any portion of the images and any portion of the derived data for commercial purposes.

3. You agree **not to** further copy, publish or distribute any portion of the SingingHead dataset to any third party for any purpose. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

4. Shanghai Jiao Tong University reserves the right to terminate your access to the SingingHead dataset at any time.


### Overview
The SingingHead dataset is a large-scale 4D dataset for singing head animation. It contains more than 27 hours of synchronized singing video, 3D facial motion, singing
audio, and background music collected from 76 subjects. 
The video is captured in 30fps and cropped into a resolution of 1024×1024.
The 3D facial motion is represented by 59-dimensional [FLAME](https://flame.is.tue.mpg.de/) parameters (50 expression + 3 global pose + 3 neck pose + 3 jaw pose).
All the data sequences are cut into equal-length 8s segments, resulting in a total of 12196 sequences.

### Data Structure
```
SingingHead
├── train.txt
├── val.txt
├── test.txt
├── video_seqs.zip
│   ├── id0_10_0_0.mp4
│   └── ...
├── flame_seqs.zip
│   ├── id0_10_0_0.pkl
│   └── ...
├── audio_seqs.zip
│   ├── id0_10_0_0.wav
│   └── ...
└── bgm_seqs.zip
    ├── id0_10_0_0_bgm.wav
    └── ...
```


## Evaluation metrics
### 2D metrics
Organize the generation results according to the following structure, and then run `python /metrics/calculate_2d_metrics.py`. Metric calculation results for all methods will be saved to `/eval_folder/metric_results`.
```
eval_folder
├── input_audio
└── generated
    ├── gt_videos
    ├── method1_generaed_videos
    ├── method2_generaed_videos
    └── ...
```

### 3D metrics



## Citation
If you use this dataset, please consider citing
```
@article{wu2025singinghead,
  title={Singinghead: A large-scale 4d dataset for singing head animation},
  author={Wu, Sijing and Li, Yunhao and Zhang, Weitian and Jia, Jun and Zhu, Yucheng and Yan, Yichao and Zhai, Guangtao and Yang, Xiaokang},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```

## Contact
- Sijing Wu [(wusijing@sjtu.edu.cn)](wusijing@sjtu.edu.cn)
