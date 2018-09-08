# clothing retrieval based on images

## 1. procedure for similar clothing retrieval

## 2. object detection

## 3. whole image retrieval VS object proposal retrieval
- the whole image retrieval results are good when the images are simple;
- the object proposal retrieval can eliminate the influence from the background (eg. Figure 1, 2, 3, 5, 7 and 8);
- the object detection process can cause minor noise (eg. Figure 8, 9 and 12: shoes are detected for the images in which shoes are not the target objects; Figure 11: pants are detected for the image where sweater is the target object)

<div align="center">
<img src="/img/12663756.jpg" height="400" width="400">
<img src="/img/12663756_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 1. Left: whole image retrieval; Right: object proposal retrieval. (Category, prdid) = (B43071701, 12663756) </em> </p>
</div>

<div align="center">
<img src="/img/14042095.jpg" height="400" width="400">
<img src="/img/14042095_skirt.jpg" height="400" width="400"/>
<p> <em> Figure 2. (Category, prdid) = (B43030101, 14042095) </em> </p>
</div>

<div align="center">
<img src="/img/14171938.jpg" height="400" width="400">
<img src="/img/14171938_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 3. (Category, prdid) = (B43071701, 14171938) </em> </p>
</div>

<div align="center">
<img src="/img/15571732.jpg" height="400" width="400">
<img src="/img/15571732_hat.jpg" height="400" width="400"/>
<p> <em> Figure 4. Category: B43071701, prd_id: 15571732 </em> </p>
</div>

<div align="center">
<img src="/img/21537227.jpg" height="400" width="400">
<img src="/img/21537227_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 5. Category: B43030101, prd_id: 21537227 </em> </p>
</div>

<div align="center">
<img src="/img/21546877.jpg" height="400" width="400">
<img src="/img/21546877_polo_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 6. Category: B43071701, prd_id: 21546877 </em> </p>
</div>

<div align="center">
<img src="/img/25755306.jpg" height="400" width="400">
<img src="/img/25755306_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 7. Category: B43072101, prd_id: 25755306 </em> </p>
</div>

<div align="center">
<img src="/img/25755307.jpg" height="400" width="400">
<img src="/img/25755307_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 8. Category: B43072101, prd_id: 25755307 </em> </p>
</div>

<div align="center">
<img src="/img/26330643.jpg" height="400" width="400">
<img src="/img/26330643_t_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 9. Category: B43072101, prd_id: 26330643 </em> </p>
</div>

<div align="center">
<img src="/img/27187591.jpg" height="400" width="400">
<img src="/img/27187591_t_shirt.jpg" height="400" width="400"/>
<p> <em> Figure 10. Category: B43072101, prd_id: 27187591 </em> </p>
</div>

<div align="center">
<img src="/img/29272047.jpg" height="400" width="400">
<img src="/img/29272047_pants.jpg" height="400" width="400"/>
<p> <em> Figure 11. Category: B43072101, prd_id: 29272047 </em> </p>
</div>

<div align="center">
<img src="/img/30045407.jpg" height="400" width="400">
<img src="/img/30045407_sweater.jpg" height="400" width="400"/>
<p> <em> Figure 12. Category: B43072101, prd_id: 30045407 </em> </p>
</div>