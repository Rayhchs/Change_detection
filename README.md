# Change Detection
In the field of remote sensing science, change detection is one of the most important tasks, especially for urban monitoring, land development monitoring, etc.
This repository perform a change detection of building using a deep learning model composed of ConvLSTM layer [1] and U-Net [2] model.

## Dataset
This repository uses LEVIR-CD remotely sensed binary change detection dataset. 
It consists of 637 VHR bi-temporal images with 1024 x 1024 pixels.
LEVIR-CD focuses on building-related change.
In this repository, 440 pairs of images are used for training, 60 pairs are used for validation and 120 pairs are used for testing.

## Citation
Please cite if you uses their dataset.

    @Article{Chen2020,
    AUTHOR = {Chen, Hao and Shi, Zhenwei},
    TITLE = {A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
    JOURNAL = {Remote Sensing},
    VOLUME = {12},
    YEAR = {2020},
    NUMBER = {10},
    ARTICLE-NUMBER = {1662},
    URL = {https://www.mdpi.com/2072-4292/12/10/1662},
    ISSN = {2072-4292},
    DOI = {10.3390/rs12101662}
    }
## Getting Started
* Clone the repository
        
      $ git clone https://github.com/Rayhchs/Change-detection.git
        
* You could download the dataset from: https://justchenhao.github.io/LEVIR/.
After that, unzip the file to train, test and val folder.

* For Training

      $ python train.py
        
* For Testing

      $ python prediction.py

## Results


## References
[1] Shi, X., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. arXiv preprint arXiv:1506.04214.

[2] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.