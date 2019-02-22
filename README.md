---
typora-root-url: resources
---

# nightShot Camera

This is a CNN based program that enhance the photo quality in low light environment which (may) works in a similar manner as Google Camera.

Given an amplification factor (gain), this program will convert the input, which is a Raw sensor data (.png format) with standard RGBG Bayer filter array, to a demosaic image (.jpg format) with similar visual quality of a long exposure photo.

| built-in post processing | output of the network |
| :----------------------: | :-------------------: |
|      ![01](/01.jpg)      |    ![02](/02.jpg)     |

For more technical details about the structure and training datasets, please refer to the original paper:

*Chen, Chen, et al. "Learning to see in the dark." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.*

This project has been made to a stand-alone windows executable file, download the [executable](https://drive.google.com/file/d/15g5ivbNkQ4XvPE5NyTNgCWteQVPqvvit/view?usp=sharing) together with the trained weights if you just want to see the result, make sure that the executable (.exe), the trained [weights](https://drive.google.com/open?id=10digkmC8MFtE5h0KlWWSfyNtdS4GH2df) (.pth), and all RAW images (.png) should be placed in the same directory.

Note: this project has not been optimized, inference may require a lot of memory, the stand-alone program has been tested with 8GB RAM without error.



### Train the model from scratch

To train the network from scratch, you need to download the dataset from the authors' [github](https://github.com/cchen156/Learning-to-See-in-the-Dark) and change the path "data_root" in config.ipynb to your own dataset. There are two sensor types in the original dataset, however only the Sony dataset is used in this project.

The intermediate weights will be saved in "saves" folder, if you want to continue the training after a pause, change the path of "save_root" in config.ipynb to "./saves" and the script will load the latest weight file before training.

To start training, simply run train.ipynb in Jupyter Notebook, you might need to change the port number of Visdom.

The file test.ipynb is the source code of the compiled stand-alone, I use pyinstaller to convert this script to a windows executable.

All scripts are tested in Jupyter Notebook with pyTorch 1.0.0 and Ubuntu 16.04.

