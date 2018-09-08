# clothing retrieval based on images

## 1. procedure for similar clothing retrieval
- use the YOLO-V3 model to crop the target clothing from original images;
- use the MobileNet model to extract the representative features (1024 dims) of the target clothing;
- apply the cosine similarity to retrieve the similar clothing.

<div align="center">
<img src="/img/similar retrieval diagram.jpg" height="400" width="500">
<p> <em> Figure 1. Diagram of the clothing retrieval procedure </em> </p>
</div>

## 2. object detection
- new label 1324 clothing images: <em>image statistics: {'shoes': 466, 't_shirt': 198, 'skirt': 145, 'glove': 43, 'blouse': 74, 'sweater': 195, 'pants': 176, 'coat': 134, 'dress': 159, 'polo_shirt': 200, 'hat': 246, 'shirt': 164}</em>;

- fine tuning of the YOLO-V3 model: <em>according to [keras-yolo3](https://github.com/qqwweee/keras-yolo3)</em>;

- results analysis.
  * download top 10 best selling categories of the GS clothing (pics, ~ GB);

| category | total number | label acc | bndbox acc |
| -------  | ------- | ------- | ------- |
| t-shirt  | 205660  | | |
| pants | 142929 | | |
| bra/panty set | 16527 | | |
| blouse | 44195 | | |
| dress | 72586 | | |
| panties | 30178 | | |
| socks | 16851 | | |
| swimsuit | 54659 | | |
| jacket | 47978 | | |
| shirt | 41698 | | |
| cardigan | 15979 | | |
| knit/sweater | 33487 | | |
| skirt | 28548 | | |
 


## 3. whole image retrieval VS object proposal retrieval
- the whole image retrieval results are good when the images are simple;
- the object proposal retrieval can eliminate the influence from the background (eg. Figure 1, 2, 3, 5, 7 and 8);
- the object detection process can cause minor noise (eg. Figure 8, 9 and 12: shoes are detected for the images in which shoes are not the target objects; Figure 11: pants are detected for the image where sweater is the target object);
- the object detection does not distinct female/male clothing (eg. Figure 6, 7, 10).

<div align="center">
<img src="/img/12663756.jpg" height="400" width="400">
<img src="/img/12663756_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 1. Left: whole image retrieval; Right: object proposal retrieval. (category, prdid) = (B43071701, 12663756) </em> </p>
</div>

<div align="center">
<img src="/img/14042095.jpg" height="400" width="400">
<img src="/img/14042095_skirt.jpg" height="400" width="400"/>
<p> <em> Figure 2. (category, prdid) = (B43030101, 14042095) </em> </p>
</div>

<div align="center">
<img src="/img/14171938.jpg" height="400" width="400">
<img src="/img/14171938_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 3. (category, prdid) = (B43071701, 14171938) </em> </p>
</div>

<div align="center">
<img src="/img/15571732.jpg" height="400" width="400">
<img src="/img/15571732_hat.jpg" height="400" width="400"/>
<p> <em> Figure 4. (category, prdid) = (B43071701, 15571732) </em> </p>
</div>

<div align="center">
<img src="/img/21537227.jpg" height="400" width="400">
<img src="/img/21537227_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 5. (category, prdid) = (B43030101, 21537227) </em> </p>
</div>

<div align="center">
<img src="/img/21546877.jpg" height="400" width="400">
<img src="/img/21546877_polo_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 6. (category, prdid) = (B43071701, 21546877) </em> </p>
</div>

<div align="center">
<img src="/img/25755306.jpg" height="400" width="400">
<img src="/img/25755306_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 7. (category, prdid) = (B43072101, 25755306) </em> </p>
</div>

<div align="center">
<img src="/img/25755307.jpg" height="400" width="400">
<img src="/img/25755307_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 8. (category, prdid) = (B43072101, 25755307) </em> </p>
</div>

<div align="center">
<img src="/img/26330643.jpg" height="400" width="400">
<img src="/img/26330643_t_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 9. (category, prdid) = (B43072101, 26330643) </em> </p>
</div>

<div align="center">
<img src="/img/27187591.jpg" height="400" width="400">
<img src="/img/27187591_t_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 10. (category, prdid) = (B43072101, 27187591) </em> </p>
</div>

<div align="center">
<img src="/img/29272047.jpg" height="400" width="400">
<img src="/img/29272047_pants.jpg" height="400" width="400"/>
<p> <em> Figure 11. (category, prdid) = (B43072101, 29272047) </em> </p>
</div>

<div align="center">
<img src="/img/30045407.jpg" height="400" width="400">
<img src="/img/30045407_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 12. (category, prdid) = (B43072101, 30045407) </em> </p>
</div>
