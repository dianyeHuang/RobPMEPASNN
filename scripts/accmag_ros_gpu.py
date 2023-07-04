#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2023-02-18 20:55:57
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2023-07-04 17:22:09
Description: 
    This script creates a node that processes the captured 
    ultrasound image and output the pulsation map image
        - Subscribe: 
            'frame_grabber/us_img' from the framegrabber node
        - Publish: 
            'usvideo_magnify/magnify_image'
            'usvideo_magnify/elapsed_time'
'''

import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from accmag_main_gpu import AccMagCore

class MagAccRos:
    def __init__(self, usimg_topic='frame_grabber/us_img'):
        # init ros node
        rospy.init_node('mag_acc_node', anonymous=True)
        
        # subscribe ultrasound images from framegrabber
        self.usimg = None
        self.flag_update = False
        self.bridgeC = CvBridge()
        self.usimg_topic_name = usimg_topic
        rospy.Subscriber(
            self.usimg_topic_name,
            Image, self.usimg_cb, queue_size=1
        )
        print('<MagAccRos> waiting for ros topic: '+self.usimg_topic_name)
        rospy.wait_for_message(self.usimg_topic_name, Image)
        print('<MagAccRos> topic received')
        
        # publish magnification result and time elapsed
        self.pub_magimg  = rospy.Publisher(
            'usvideo_magnify/magnify_image', 
            Image, queue_size=1
        )
        self.pub_magtime = rospy.Publisher(
            'usvideo_magnify/elapsed_time', 
            Float32, queue_size=1
        )
        
        # initialize acceleration magnification
        self.accmag = AccMagCore(
            fps=30.0,
            freq_des=1.55,
            freq_range=[0.9, 1.9],
            butter_order=3.0,
            # gamma=0.65 # for visualization radial
            gamma=0.80 # for visualization carotid
        )
        # - initialize DoG and butterworth filters
        print('<MagAccRos> loading filters to gpu ...')
        self.accmag.magnify(
            img=self.usimg,
            img_scale=None,
            mag_factor=38.0,
            flag_vis=False
        )
        print('<MagAccRos> done!')
        
        self.flag_update=False
    
    def usimg_cb(self, msg:Image):
        if not self.flag_update:
            self.flag_update = True
            self.usimg = self.bridgeC.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            # test received results
            # cv2.imshow('received images', self.usimg)
            # cv2.waitKey(1)
            
    def run(self, 
            loop_hz=30.0, 
            img_scale=None,
            mag_factor=20.0):
        
        seq = 0
        msg_float = Float32()
        
        rate = rospy.Rate(loop_hz)
        while not rospy.is_shutdown():
            rate.sleep()
            if self.flag_update:
                # magnify ultrasound image
                mag_img, elapsed_t = self.accmag.magnify(
                    img=self.usimg,
                    img_scale=img_scale,
                    mag_factor=mag_factor,
                    flag_vis=False
                )
                
                # - publish result image
                msg_img = self.bridgeC.cv2_to_imgmsg(
                    # mag_img # single image
                    np.hstack((self.usimg.copy(), mag_img))
                )
                msg_img.header.frame_id = 'us_probe'
                msg_img.header.seq = seq
                seq += 1
                msg_img.header.stamp = rospy.get_rostime()
                self.pub_magimg.publish(msg_img)
                
                # - publish elapsed time (ms)
                msg_float.data = elapsed_t*1e+3
                self.pub_magtime.publish(msg_float)
                
                # visualize results
                # cv2.imshow('mag res', mag_img)
                # cv2.waitKey(1)
                self.flag_update=False
    
if __name__ == '__main__':
    magros = MagAccRos(usimg_topic='frame_grabber/us_img')
    magros.run(
        loop_hz=30.0, 
        img_scale=None,
        mag_factor=12.0 # 38.0 for radial, 12.0 for carotid
    )
