# 深度学习Demo：基于自编码器(U-Net)的人脸场景转换

本实验主要使用自编码器完成人脸变换，主要用到的网络为[U-Net](https://arxiv.org/pdf/1505.04597.pdf)，以及参考代码[datitran/face2face-demo](https://github.com/datitran/face2face-demo)。如下图所示，网络的输入为人脸的landmark图，输出为人脸图。

![img](https://github.com/zjutcsai/deeplearning/blob/master/001_face2face/data.jpg)

## 文件结构图

- 001_face2face(子工程目录)
	- data(训练及测试数据文件夹, github已忽略)
		- origin(demo数据的数据集)
			- train: 训练集
			- val: 验证集
	- lib(库文件夹)
		- nets(网络拓扑文件夹，可自行在该文件夹中添加新的网络拓扑)
			- layers.py: 自定义层，包含卷积、反卷积、激活、正则等操作
			- pix2pix.py: 网络模型1，基于U-Net的Pix2Pix网络
			- pix2pix_gan.py: 网络模型2，基于U-Net的Pix2Pix网络，并使用GAN来进行对抗训练
		- dataset.py: 数据操作文件，包含载入训练&测试数据、batch采样等操作
		- utils.py: 工具函数文件，包含网络输入预处理、网络输出后处理、图像保存等函数
		- generate_data.py: 原始视频图像处理函数，得到脸部landmark图
	- model(模型保存文件夹, github已忽略)
		- Pix2Pix(网络模型1保存文件夹)
		- Pix2PixGAN(网络模型2保存文件夹)
	- output(模型验证文件夹, github已忽略)
		- Pix2Pix(网络模型1验证保存文件夹)
		- Pix2PixGAN(网络模型2验证保存文件夹)
	- tools(脚本文件夹)
		- train_script.py: 训练脚本
		- test_script.py: 测试脚本
	- main.py: 主函数入口

## 程序参数设置

参数设置全部在main.py中完成，主要包含：

- max_step:    网络最大训练次数，默认80000
- save_step:   模型保存间隔，默认每训练1000次保存一次模型
- vis_step:    网络验证检测，默认每训练200次验证一次模型合成效果
- log_step:    网络训练打印间隔，模型每训练20次打印误差等信息
- vis_dir:     网络验证阶段输出图片保存文件夹，默认保存在output/{net.name}/
- model_dir:   网络模型参数保存文件夹，默认保存在model/{net.name}/
- train_dir:   训练数据文件夹，默认为data/origin/train
- test_dir:    验证数据文件夹，默认为data/origin/val
- batch_size:  网络输入batch大小，默认16
- image_size:  网络输入尺寸大小，默认输入图像宽高为256
- net_name:    使用网络模型名称，默认Pix2Pix，目前版本还额外提供Pix2PixGAN
- mode:        程序运行模式，默认为训练train，也可以使用测试模式test
- test_video:  测试视频路径

## 程序运行环境与依赖库

程序可以在Windows, Macos以及Linux环境下运行，使用的语言为Python 3.x，运行所需要的依赖库包含:

- Python 3.x: Windows用户建议直接安装[anaconda](https://www.anaconda.com/download/)
- [Tensorflow](https://www.tensorflow.org/install/): 建议使用最新版本1.7以上
- [Opencv](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv): 选择正确的版本号，`opencv_python‑3.4.1+contrib‑cp36‑cp36m‑win_amd64.whl`其中3.4.1为opencv版本号;contrib为opencv贡献包，包含了额外的工具(可不选);cp36-cp36m为python3.6版本；win_amd64为64位系统版本，win32为32位系统版本，请正确选择
- dlib: Windows用户建议安装vs2017, 然后使用vs自带的命令提示工具`pip install dlib`即可

## 程序运行指令

1. 训练自己的网络

训练最好使用GPU，否则训练速度会比较慢，训练前可以在main.py修改上述参数，也可以在控制台传入参数，运行方式为：

```
python main.py --mode train \
               --max-step 50000 \
	       --save-step 2000 \
	       --net-name Pix2Pix
```

2. 测试自己的网络

当训练结束后，就可以测试自己的模型，测试需要传入一段测试视频，当然也可以传入图片(需要自己写对应的函数)，运行方式为：

```
python main.py --mode test \
               --test-video test_video.mp4
```

为了便于用户测试模型，我们将提供训练完毕的模型以及数据，下载地址为：[face2face预训练模型]()

## 实验结果展示

![img](https://github.com/zjutcsai/deeplearning/blob/master/001_face2face/result.jpg)

上述图片中有三张图片，最左边为视频输入图像，中间的为`generate_data`处理过后的网络输入landmark图，最后边的为对应的网络输出图。
