# Change Detection
In the field of remote sensing science, change detection is one of the most important tasks, especially for urban monitoring, land development monitoring, etc.
This repository perform a change detection of building using a deep learning model composed of ConvLSTM layer [1] and U-Net [2] model.


## Dataset
This repository uses LEVIR-CD remotely sensed binary change detection dataset. 
It consists of 637 VHR bi-temporal images with 1024 x 1024 pixels.
LEVIR-CD focuses on building-related change.
In this repository, 440 pairs of images are used for training, 60 pairs are used for validation and 120 pairs are used for testing.

| Pre Image  | Post Image | Label |
| ------------- | ------------- |------------- |
| <img src="https://github.com/Rayhchs/Change_detection/blob/main/train/A/train_1.png" alt="Editor" width="250" title="Pre"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/train/B/train_1.png" alt="Editor" width="250" title="Post"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/train/label/train_1.png" alt="Editor" width="250" title="Label"> |


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
Here we display some well detected results. 
In region 7, change detection result can achieve **0.84** of IOU and **0.91** of F1-score; In region 10, change detection result can achieve **0.83** of IOU and **0.90** of F1-score; In region 14, change detection result can achieve **0.83** of IOU and **0.90** of F1-score. 
However, the 120 testing pairs can only achieve **0.66** of mIOU and **0.75** of mF1-score.

### Area 7
| Pre Image  | Post Image | Label | Detection Result |
| ------------- | ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/A/test_7.png" alt="Editor" width="250" title="Pre"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/B/test_7.png" alt="Editor" width="250" title="Post"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/label/test_7.png" alt="Editor" width="250" title="Label"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/predict/7.png" alt="Editor" width="250" title="Predict"> |

### Area 10
| Pre Image  | Post Image | Label | Detection Result |
| ------------- | ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/A/test_10.png" alt="Editor" width="250" title="Pre"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/B/test_10.png" alt="Editor" width="250" title="Post"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/label/test_10.png" alt="Editor" width="250" title="Label"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/predict/10.png" alt="Editor" width="250" title="Predict"> |

### Area 14
| Pre Image  | Post Image | Label | Detection Result |
| ------------- | ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/A/test_14.png" alt="Editor" width="250" title="Pre"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/B/test_14.png" alt="Editor" width="250" title="Post"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/label/test_14.png" alt="Editor" width="250" title="Label"> | <img src="https://github.com/Rayhchs/Change_detection/blob/main/test/predict/14.png" alt="Editor" width="250" title="Predict"> |


## Citation
Please cite this paper if you uses their dataset.

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


## References
[1] Shi, X., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. arXiv preprint arXiv:1506.04214.

[2] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
