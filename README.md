Unsupervised Dense Nuclei Detection and Segmentation with Prior Self-activation Map For Histology Images
====
Training
------
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
![](https://gitee.com/cpystan/prior_-self-activation_-map/raw/master/pics/fig.jpg =400x300)

### Train NDN
```
python main_monuseg.py --mode 'train_second_stage' 
```

### Generate auxiliary labels
```
python main_monuseg.py --mode 'generate_voronoi' --model 'model_path'
```

### Train NSN
```
python main_monuseg.py --mode 'train_final_stage' 
```

### Test
```
python main_monuseg.py --mode 'test' --model 'model_path'
```