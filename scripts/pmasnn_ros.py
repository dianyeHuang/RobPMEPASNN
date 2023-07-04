#!/usr/bin/env python 
'''
Author: Dianye Huang
Date: 2023-02-23 19:11:30
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2023-07-04 17:36:51
Description: 
    This script creates a node that serves as a ros_bridge to 
    load the PAS-NN and publish the segmentation results.
        - Subscribe:
            - '/usvideo_magnify/magnify_image' from accmag_ros_gpu.py
        - Publish:
            - '/usvideo_magnify/seg_result' 
'''

import torch
import cv2
from MagNet import MagNet, MagNet_v2, AttUNet
import numpy as np
from torchvision import transforms

import time
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class PMASNN_rosinterface:
    def __init__(self, 
                model_path='',
                sub_topic_name='/usvideo_magnify/magnify_image',
                pub_topic_name='/usvideo_magnify/seg_result'
        ):
        # ros init
        # -- ros subscribe images
        rospy.init_node('PMAS_NN_node', anonymous=True)
        rospy.Subscriber(
            sub_topic_name, 
            Image, 
            self.img_cb,
            queue_size=1
        )
        self.usimg  = None
        self.magimg = None
        self.wsplit = None
        self.imresize = None
        self.flag_update = False
        self.bridgeC  = CvBridge()
        print('<PMASNN_ros> waiting for rostopic: ', sub_topic_name)
        rospy.wait_for_message(sub_topic_name, Image)
        print('<PMASNN_ros> topic received!')
        
        # -- ros publish images
        self.pub_pred = rospy.Publisher(
            pub_topic_name, 
            Image,
            queue_size=1
        )
        
        # Model init
        self.transform_image = transforms.Normalize(0.5, 0.5) 
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print('<PMASNN_ros> loading PMAS-NN model ...')
        
        self.model = MagNet_v2(init_features=64).to(self.device)
        self.model.load_state_dict(
            torch.load(
                model_path, 
                map_location=self.device 
            )['state_dict']
        )
        # self.model.eval()  # disable gradient tracking
        self.model.train()  # disable gradient tracking
        
        torch.cuda.empty_cache()
        print('<PMASNN_ros> PMAS-NN initialization done!')
    
    def img_cb(self, msg):
        if not self.flag_update:
            tmp_img = self.bridgeC.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            if self.wsplit is None:
                self.wsplit   = int(tmp_img.shape[1]/2)
                self.imresize = [self.wsplit, tmp_img.shape[0]] # for resize prediction result
                
            self.usimg  = tmp_img[:, :self.wsplit].astype(np.float64)/255.0
            self.magimg = tmp_img[:, self.wsplit:].astype(np.float64)/255.0
            self.flag_update = True
            
            # for debug
            # print(type(self.usimg[0, 0]))
            # print(type(self.magimg[0, 0]))
            # print('shape of  usimg: ', self.usimg.shape)
            # print('shape of magimg: ', self.magimg.shape)
            # cv2.imshow('us image', np.uint8(self.usimg))
            # cv2.imshow('pm image', np.uint8(self.magimg))
            # cv2.waitKey(1)
    
    def run(self, loop_hz=30.0):
        img_seq  = 0
        rate = rospy.Rate(loop_hz)
        while not rospy.is_shutdown():
            rate.sleep()
            if self.flag_update:
                # start_time = time.time()
                # convert image size
                self.usimg  = cv2.resize(
                            self.usimg, (256, 256),
                            interpolation=cv2.INTER_LANCZOS4
                        )
                self.magimg = cv2.resize(
                            self.magimg, (256, 256),
                            interpolation=cv2.INTER_LANCZOS4
                        )
                # convert input into GPU
                x = self.transform_image(
                        torch.from_numpy(
                                self.usimg
                            ).float().to(
                                self.device
                        ).view(-1, 256, 256)
                    ).view(-1, 1, 256, 256)
                m = self.transform_image(
                        torch.from_numpy(
                                self.magimg
                            ).float().to(
                                self.device
                        ).view(-1, 256, 256)
                    ).view(-1, 1, 256, 256)
                # compute prediction and resize
                pred = self.model(
                            x, m
                        ).view(256,256).cpu().detach().numpy()
                pred = cv2.resize(pred, self.imresize)
                
                self.flag_update=False
                
                # publish prediction result
                msg_img = self.bridgeC.cv2_to_imgmsg(
                    pred*255.0
                )
                msg_img.header.frame_id = 'us_probe'
                msg_img.header.seq = img_seq
                img_seq += 1
                msg_img.header.stamp = rospy.get_rostime()
                self.pub_pred.publish(msg_img)
                
                # for debug purpose 
                # end_time = time.time()
                # print('Time ellapse: %.2f ms'%((end_time-start_time)*1e+3))
                # cv2.imshow('pred_res', np.uint8(255*pred))
                # cv2.waitKey(1) 
    

if __name__ == '__main__':
    best_path = '<path_to_>/model_best_carotid.pth' # for carotid artery
    # best_path = '<path_to_>/model_best_radial.pth' # for radial artery
    pmas_ros  = PMASNN_rosinterface(model_path=best_path)
    pmas_ros.run()
