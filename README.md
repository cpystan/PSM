PSM
====
Exploring Label-free Cell Recognition with Prior Self-activation Maps， MICCAI 2023
------
 <img src="https://github.com/cpystan/PSM-MICCAI/blob/master/pics/framework.png" width = "80%" height = "80%" alt="framework" align=center />

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
### Generate Pseudo Masks
Following the self-supervised training, we can then obtain the pseudo masks using the pre-trained model.
```
python main_monuseg.py --mode 'generate_label' --model 'model_path'
```

 <img src="https://github.com/cpystan/Prior-Self-activation-Map/blob/master/pics/fig.jpg" width = "60%" height = "60%" alt="self-activation map vs. pseudo mask" align=center />
 
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
