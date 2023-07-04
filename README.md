
# PAS-NN for Robotic Ultrasound

#### Motion Magnification in Robotic Sonography: Enabling Pulsation-Aware Artery Segmentation

This work presents a novel pulsation-assisted segmentation neural network (PAS-NN) by explicitly taking advantage of the cardiac-induced motions. Motion magnification techniques are employed to amplify the subtle motion within the frequency band of interest to extract the pulsation signals from sequential US images. 
<div align="center">
<img src=asset/fig_introduction.jpg  width=60% title=asadsds/>
</div>
A robotic arm is utilized to acquire the ultrasound images stably. 
<div align="center">
<img src=asset/fig_imaging_pipeline.jpg  width=50% title=asadsds/>
</div>

This repository includes codes for implementing the pulsation-map algorithm and the pulsation-assisted segmentation neural network (PAS-NN) architecture. 

- Steps for implementing the pulsation-map algorithm
<div align="center">
<img src=asset/fig_pme_pipeline.jpg  width=40%/>
</div>

- The PAS-NN consists of two decoder and a decoder. The pulsation guidance information is integrated into the network by the skip connection and attantion gates mechanisms.
<div align="center">
<img src=asset/fig_network_structure.png  width=80%/>
</div>

## Video Demo
<center> <iframe width="560" height="315" src="https://www.youtube.com/embed/cLeN-TGS1f8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></center>


## Citation
If you found this work interesting and adopted part of it to your own research, or if this work inspires your research, you can cite our paper by:

```
@inproceedings{pasnn23,
  title     = {Motion Magnification in Robotic Sonography: Enabling Pulsation-Aware Artery Segmentation},
  author    = {Dianye, Huang and
               Yuan, Bi and
               Nassir, Navab and
               Zhongliang, Jiang},
  booktitle = {IEEE/JRS Conference on Intelligent Robots and Systems (IROS)},
  year = {2023}
}
```
