PSM
====
Exploring Unsupervised Cell Recognition with Prior Self-activation Maps， MICCAI 2023
------
See our paper [here](https://arxiv.org/abs/2308.11144) 

 <img src="https://github.com/cpystan/PSM/blob/master/pics/framework.png" width = "80%" height = "80%" alt="framework" align=center />

**MoNuSeg dataset**

Kumar, N., Verma, R., Sharma, S., Bhargava, S., Vahadane, A., Sethi, A.: A
dataset and a technique for generalized nuclear segmentation for computational
pathology.

**BCData**

Huang, Z., Ding, Y., Song, G., Wang, L., Chen, J.: BCData: A Large-Scale Dataset
and Benchmark for Cell Detection and Counting.

Arrange your MoNuSeg dataset:
 
```
data
-training
---Annotations
----1.png
----2.png
.......
---Tissue Images
----1.png
----2.png
.......

-Testing
----1.png
----2.png
.......
```

### Self-supervised Training
We provide several self-supervised learning methods.
```
python main_monuseg.py --mode 'train_base'    #Similarity Measure
python main_monuseg.py --mode 'train_contrastive'    #Basic Contrastive learning
python main_monuseg.py --mode 'train_random_rotate'    #Predicting rotate
python main_monuseg.py --mode 'train_simsiam'    #Simsiam
python main_monuseg.py --mode 'train_mean_value'    #Predicting average pixel
```

Checkpoint of our trained self-supervised network is provided:

[Checkpoint trained by train_base](https://pan.baidu.com/s/1mzMGe3vSqiWgftWCaM2atg?pwd=abcd)  

### Generate Pseudo Masks
Following the self-supervised training, we can then obtain the pseudo masks using the pre-trained model.
```
python main_monuseg.py --mode 'generate_label' --model 'model_path'
```

 <img src="https://github.com/cpystan/PSM/blob/master/pics/fig.jpg" width = "60%" height = "60%" alt="self-activation map vs. pseudo mask" align=center />
 
demo of the PSM and its pseudo label after clustering

### Generate Voronoi Labels
```
python main_monuseg.py --mode 'train_second_stage'  # train a network to get the point prediction
python main_monuseg.py --mode 'generate_voronoi' --model 'model_path' $ get voronoi labels
```

### Train Segmentation Network
```
python main_monuseg.py --mode 'train_final_stage' 
```

### Test
```
python main_monuseg.py --mode 'test' --model 'model_path'
```

### Citation
```
@InProceedings{psm,
author="Chen, Pingyi
and Zhu, Chenglu
and Shui, Zhongyi
and Cai, Jiatong
and Zheng, Sunyi
and Zhang, Shichuan
and Yang, Lin",
title="Exploring Unsupervised Cell Recognition with Prior Self-activation Maps",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="559--568"
}

```
