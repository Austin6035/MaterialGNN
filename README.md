### 项目简介

基于图神经网络的材料性质预测

### Requirements:

python3

PyTorch 1.9.0-cu10.1

PyTorch Geometric 2.0.3

Obabel

Pymatgen

### How to install:

- git clone http://ip:port/xxx

- 下载zip并解压

### How to prepare data:

将材料的结构文件和性质id_prop.csv文件放到同一目录下，放到dataset/目录下

**note：**id_prop.csv中第一列material_id需要和结构文件xxx.xyz中文件名对应(不包含后缀)，但同一个数据集下仅支持一种类型的结构文件(xyz, cif, mol, sdf)，程序会自动判断文件后缀名，通常晶体为cif文件，而其他都为分子文件。

```
|- dataset
	|- custom_dataset
		|- xxx.xyz
		|- ...
		|- id_prop.csv
```

### How to train model:

```shell
python train.py dataset/custom_dataset
```

在train.py中可查看参数设置并指定，或通过命令行传入

```shell
python train.py dataset/custom_dataset optim=Adam
```

### How to predict:

对于预测功能，需要指定预训练的模型，位于weight/目录下

```shell
python predict.py weight/pretrained_model.pth.tar custom_dataset
```

Notes:
The data related to "Deep Learning Accelerated Discovery of Metastable Iridium Dioxides for the Oxygen Evolution Reaction" can be accessed in the folder "Application", whose contents are listed below:
a."C2DB dataset"--The C2DB dataset, its corresponding model, and its training results.
b."MP dataset"--The MP dataset, its corresponding model, and its training results.
c."IrO2 train"--The IrO2 dataset, its corresponding model, and its training results of the 3000 initail structures(n = 1, 2).
d."IrO2 predict"--The IrO2 dataset and its predicting results of the 7000 complex structures(n = 3, 4, 5, 6, 7).