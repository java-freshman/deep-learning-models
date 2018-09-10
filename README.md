# A. Instruction

## 1. Package 
- object_detection: a modulus to detect the target object in given images.
Modified based on the [YOLO-V3](https://github.com/qqwweee/keras-yolo3)
- feature_extraction: a modulus to extract the image representative features.
Modified based on the [Trained image classification models for Keras](https://github.com/fchollet/deep-learning-models)

## 2. Usage

```
step 1: image crop
  - upload the original images to dir=""
  - $python3 img_crop.py --new_dcd=""
  - output to dir=""
step 2: feature extraction
  - $python3 img_feature_extraction.py --new_dcd=""
  - output to dir=""
step 3: similar retrieval
  - $python3 img_similarity_calc.py
  - output to dir=""
step 4: output retrieval samples
  - $jupyter notebook
  - run ipynb scripts
```

# B. Clothing Retrieval Example

## 1. Retrieval Procedure
- use the YOLO-V3 model to crop the target clothing from original images;
- use the MobileNet model to extract the representative features (1024 dims) of the target clothing;
- apply the cosine similarity to retrieve the similar clothing.

<div align="center">
<img src="/src/similar retrieval diagram.jpg" height="400" width="500">
<p> <em> Figure 1. Diagram of the clothing retrieval procedure </em> </p>
</div>

## 2. Object Detection
- new label 1324 clothing images: <em>image statistics: 
{'shoes': 466, 't_shirt': 198, 'skirt': 145, 'glove': 43, 
'blouse': 74, 'sweater': 195, 'pants': 176, 'coat': 134, 
'dress': 159, 'polo_shirt': 200, 'hat': 246, 'shirt': 164}</em>;

- fine tuning of the YOLO-V3 model: <em>according to 
[keras-yolo3](https://github.com/qqwweee/keras-yolo3)</em>;

- results analysis.
  * crawl top 10 best sale categories of the GS clothing 
  (in total 641,262 pics);
  * trained YOLO-V3 model performance: 
  the label ACC is not good, however we care more about 
  the bndbox because what we need in current project is 
  to crop the target objects out of the original images 
  correctly. From the following table we can see that 
  the bndbox ACC is high for all available categories 
  except the dress category. After carefully analysis, 
  this is mostly due to confusion between the skirt and 
  dress. <em> (Note that for the cases the model mistakes between 
  the shirt, t-shirt, polo-shirt, or blouse, we still consider
  the bndbox is correct because of all of them locate at
  the same position of the body. But the dress category
  is the one we need to improve next.)</em>
  
| category | image number | label ACC | bndbox ACC |
| :-------:  | :-------: | :-------: | :-------: |
| t-shirt  | 205660 | 64.6% | 97.1% |
| pants | 142929 | 76.9% | 93.1% |
| bra/panty set | 16527 | NA | NA |
| blouse | 44195 | 26.7% | 98.4% |
| dress | 72586 | 49.56% | 49.56% |
| panties | 30178 | NA | NA |
| socks | 16851 | NA | NA |
| swimsuit | 54659 | NA | NA |
| shirt | 41698 | 90.7% | 97.1% |
| cardigan | 15979 | 21.7% | 96.% |


## 3. Retrieval Results
- whole image retrieval:
  * results are only good when the images are simple
- object proposal retrieval:
  * eliminate the influence from the background 
  (eg. Figure 1, 2, 3, 5, 7 and 8);
  * YOLO-V3 crop process causes minor noise 
  (eg. Figure 8, 9 and 12: shoes are detected for the 
  images in which shoes are not the target objects; 
  Figure 11: pants are detected for the image where 
  sweater is the target object);
  * YOLO-V3 model does not distinct female/male clothing 
  (eg. Figure 6, 7, 10).

<div align="center">
<img src="/src/12663756.jpg" height="400" width="400">
<img src="/src/12663756_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 1. Left: whole image retrieval; Right: object proposal retrieval. (category, prdid) = (B43071701, 12663756) </em> </p>
</div>

<div align="center">
<img src="/src/14042095.jpg" height="400" width="400">
<img src="/src/14042095_skirt.jpg" height="400" width="400"/>
<p> <em> Figure 2. (category, prdid) = (B43030101, 14042095) </em> </p>
</div>

<div align="center">
<img src="/src/14171938.jpg" height="400" width="400">
<img src="/src/14171938_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 3. (category, prdid) = (B43071701, 14171938) </em> </p>
</div>

<div align="center">
<img src="/src/15571732.jpg" height="400" width="400">
<img src="/src/15571732_hat.jpg" height="400" width="400"/>
<p> <em> Figure 4. (category, prdid) = (B43071701, 15571732) </em> </p>
</div>

<div align="center">
<img src="/src/21537227.jpg" height="400" width="400">
<img src="/src/21537227_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 5. (category, prdid) = (B43030101, 21537227) </em> </p>
</div>

<div align="center">
<img src="/src/21546877.jpg" height="400" width="400">
<img src="/src/21546877_polo_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 6. (category, prdid) = (B43071701, 21546877) </em> </p>
</div>

<div align="center">
<img src="/src/25755306.jpg" height="400" width="400">
<img src="/src/25755306_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 7. (category, prdid) = (B43072101, 25755306) </em> </p>
</div>

<div align="center">
<img src="/src/25755307.jpg" height="400" width="400">
<img src="/src/25755307_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 8. (category, prdid) = (B43072101, 25755307) </em> </p>
</div>

<div align="center">
<img src="/src/26330643.jpg" height="400" width="400">
<img src="/src/26330643_t_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 9. (category, prdid) = (B43072101, 26330643) </em> </p>
</div>

<div align="center">
<img src="/src/27187591.jpg" height="400" width="400">
<img src="/src/27187591_t_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 10. (category, prdid) = (B43072101, 27187591) </em> </p>
</div>

<div align="center">
<img src="/src/29272047.jpg" height="400" width="400">
<img src="/src/29272047_pants.jpg" height="400" width="400"/>
<p> <em> Figure 11. (category, prdid) = (B43072101, 29272047) </em> </p>
</div>

<div align="center">
<img src="/src/30045407.jpg" height="400" width="400">
<img src="/src/30045407_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 12. (category, prdid) = (B43072101, 30045407) </em> </p>
</div>

## 4. Reference
- [He ACCV16] Fast fashion guided clothing image retrieval: delving deeper into what feature makes fashion
- [Kota ICCV13] Paper doll parsing: Retrieving similar styles to parse clothing items
- [Wang CVPR14] Learning fine-grained image similarity with deep ranking
- [Kiapour ICCV15] Where to buy it: matching street clothing photos in online shop
- [Chen ECCV12] Describing clothing by semantic attributes
- [Redmon arXiv18] YOLOv3: An incremental improvement  
- [FashionAI 服装属性标签识别竞赛](https://blog.csdn.net/weixin_38243861/article/details/80900796)  
- [DeepFashion 服装公开数据集概述](https://blog.csdn.net/fuwenyan/article/details/78203803)  
- [服装分类检索识别数据集](https://blog.csdn.net/fuwenyan/article/details/79224753)  
- [Where to buy it?](http://tamaraberg.com/street2shop/)  
- [CNN-目标检测、定位、分割](https://blog.csdn.net/MyArrow/article/details/51878004)  
- [基于深度学习的目标检测算法-YOLO](https://blog.csdn.net/u013989576/article/details/72781018)  
- [YOLOv3 训练自己的数据集](https://blog.csdn.net/zzhang_12/article/details/80393448)  
- [Computer vision datasets](https://handong1587.github.io/computer_vision/2015/09/24/datasets.html)  
- [Image&Vision Group](https://caiivg.weebly.com/dataset.html)  
- [Dataset: Image-net](http://www.image-net.org/)  
- [DarkNet](https://pjreddie.com/darknet/yolo/)  
- [GitHub: LabelImg](https://github.com/tzutalin/labelImg)