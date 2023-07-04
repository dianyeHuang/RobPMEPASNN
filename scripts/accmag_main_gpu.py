'''
Author: Dianye Huang
Date: 2023-02-16 14:23:24
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2023-07-04 17:32:58
Description: 
    This script implements realtime magnification of continous 
    ultrasound video streaming data. And output magnified part
    of the us images as a new input for a segmentation network.
    
    - This code is implemented with a GPU to accelerate the
    - matrix calculations
'''

import cv2
import numpy as np
from accmag_utils_gpu import (
    AccMagIO, AccMagFilter
)

import time
import torch
import pickle
from tqdm import tqdm
from pytictoc import TicToc
import matplotlib.pyplot as plt

def check_gpu():
    print('The availability of GPU: ', torch.cuda.is_available())
    print('Number of GPUs: ', torch.cuda.device_count())
    print('Device id: ', torch.cuda.current_device())
    print('Device name: ', torch.cuda.get_device_name(
        torch.cuda.current_device()
    ))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

class AccMagCore:
    def __init__(self, 
                fps=30.0,
                freq_des=1.5, 
                freq_range=[0.9, 2.5], 
                butter_order=3.0,
                gamma=1.2):
        
        # Filtering parameters
        self.fps = fps
        self.freq_des   = freq_des 
        self.freq_range = freq_range
        self.device = torch.device('cuda' if \
                        torch.cuda.is_available() else 'cpu')
        
        # DoG filter and Butterworth filter initialization
        self.afilter = AccMagFilter()
        # - DoG filter
        self.DoG_kernel = self.afilter.get_DoG_1d_kernel(
                                        freq_des=freq_des, 
                                        fps=fps, 
                                        flag_vis=False  # to plot the filter
                                    )
        self.flag_init_DoG = False
        # - Butterworth filter
        self.butter_b, self.butter_a = self.afilter.get_butter_bp_params(
                                        freq_range=freq_range,
                                        fps=fps,
                                        order=butter_order
                                    )
        
        # print('butterworth: b\n', self.butter_b)
        # print('butterworth: a\n', self.butter_a)
        
        self.butter_rnb = torch.tensor(
            np.flip(self.butter_b/self.butter_a[0],
                            axis=0)[:, None, None].copy(), 
            requires_grad=False
        ).to(self.device)
        self.butter_rna = torch.tensor(
            np.flip(self.butter_a[1:]/self.butter_a[0],
                            axis=0)[:, None, None].copy(), 
            requires_grad=False
        ).to(self.device)
        self.flag_init_butter = False
        
        # for visualization
        invGamma = 1.0 / gamma
        self.gamma_tab = np.array([((i/255.0)**invGamma)*255 for 
                            i in np.arange(0, 256)]).astype("uint8")
        
        # scale image
        self.flag_img_scale = False
        self.img_h = None
        self.img_w = None 
        self.img_sh = None # scaled image height
        self.img_sw = None # scaled image width
        
        # tictoc
        self.tictoc = TicToc()
        
    def magnify(self, 
                img, 
                img_scale=None, 
                mag_factor=20.0,
                flag_vis=False,
                flag_fast_vis=False
        ):
        start_time = time.time()
        
        # resize img 
        if img_scale is not None: 
            if not self.flag_img_scale:
                self.flag_img_scale = True
                self.img_h = img.shape[0]
                self.img_w = img.shape[1]
                self.img_sh = int(self.img_h*img_scale)
                self.img_sw = int(self.img_w*img_scale)
            img = cv2.resize(img, (self.img_sw, self.img_sh))            
        
        # - DoG filter
        if not self.flag_init_DoG:
            self.flag_init_DoG = True
            self.afilter.init_DoG_buffer(
                kernel_len=len(self.DoG_kernel),
                input_shape=img.shape
            )        
        
        img = torch.tensor(
            img, requires_grad=False
        ).to(self.device)
        
        acc_img = self.afilter.update_DoG_filter(
            kernel=self.DoG_kernel,
            img=img
        )
        
        # - Butterworth bandpass filter
        if not self.flag_init_butter:
            self.flag_init_butter = True
            self.afilter.init_butter_bp_buffer(
                b=self.butter_b, 
                a=self.butter_a, 
                input_shape=img.shape # TODO
            )
        
        amp_img = self.afilter.update_bp_filter(
            input=acc_img, # TODO acc_img : for acc component, img: for linear component
            # input=img, 
            rn_b=self.butter_rnb,
            rn_a=self.butter_rna
        )
        
        # get mag_image
        amp_img = amp_img.cpu().numpy()
        if img_scale is not None: 
            amp_img = cv2.resize(amp_img, (self.img_w, self.img_h))
        vis_img = cv2.convertScaleAbs(mag_factor*amp_img)
        vis_img = cv2.LUT(np.array(vis_img, dtype = np.uint8), self.gamma_tab)
        
        end_time = time.time()
        
        # for 
        if flag_vis:
            img = img.cpu().numpy()
            # img = cv2.convertScaleAbs(img.cpu().numpy() + mag_factor*amp_img)
            if img_scale is not None: 
                img = np.uint8(cv2.resize(img, (self.img_w, self.img_h)))
            cv2.imshow('raw and m image', np.hstack((img, vis_img)))
            if flag_fast_vis:
                cv2.waitKey(1)
            else:
                cv2.waitKey(20)
        
        return vis_img, end_time-start_time

    def magnify_log(self, 
                img, 
                img_scale=None, 
                mag_factor=20.0,
                flag_vis=False,
                flag_fast_vis=False
        ):
        start_time = time.time()
        
        # resize img 
        if img_scale is not None: 
            if not self.flag_img_scale:
                self.flag_img_scale = True
                self.img_h = img.shape[0]
                self.img_w = img.shape[1]
                self.img_sh = int(self.img_h*img_scale)
                self.img_sw = int(self.img_w*img_scale)
            img = cv2.resize(img, (self.img_sw, self.img_sh))            
        
        # - DoG filter
        if not self.flag_init_DoG:
            self.flag_init_DoG = True
            self.afilter.init_DoG_buffer(
                kernel_len=len(self.DoG_kernel),
                input_shape=img.shape
            )        
        
        img = torch.tensor(
            img, requires_grad=False
        ).to(self.device)
        
        acc_img = self.afilter.update_DoG_filter(
            kernel=self.DoG_kernel,
            img=img
        )
        
        # - Butterworth bandpass filter
        if not self.flag_init_butter:
            self.flag_init_butter = True
            self.afilter.init_butter_bp_buffer(
                b=self.butter_b, 
                a=self.butter_a, 
                input_shape=img.shape # TODO
            )
        
        amp_img = self.afilter.update_bp_filter(
            input=acc_img,
            rn_b=self.butter_rnb,
            rn_a=self.butter_rna
        )
        
        # get mag_image
        amp_img = amp_img.cpu().numpy()
        if img_scale is not None: 
            amp_img = cv2.resize(amp_img, (self.img_w, self.img_h))
        vis_img = cv2.convertScaleAbs(mag_factor*amp_img)
        vis_img = cv2.LUT(np.array(vis_img, dtype = np.uint8), self.gamma_tab)
        
        end_time = time.time()
        
        if flag_vis:
            img = img.cpu().numpy()
            # img = cv2.convertScaleAbs(img.cpu().numpy() + mag_factor*amp_img)
            if img_scale is not None: 
                img = np.uint8(cv2.resize(img, (self.img_w, self.img_h)))
            cv2.imshow('raw and m image', np.hstack((img, vis_img)))
            if flag_fast_vis:
                cv2.waitKey(1)
            else:
                cv2.waitKey(20)
        
        return vis_img, end_time-start_time, acc_img.cpu().numpy(), amp_img


def main():    
    '''
    The pickle files stores the tracked ultrasound images every one second,
    for each pickle file, it is named sequentially. The stored variable is
    of dictornary type, with keys of "ee_poses" and "us_images". The the 
    followings, the data is loaded as a spatiotemporal imageset and then
    processed according to the steps presented in the paper.
    '''
    # carotid
    filepath_dir  = 'xxxxx'
    file_prefix   = 'tmp_'
    save_dir  = 'xxxxx'
    scan_type = 'carotid'
    
    # radial
    # filepath_dir  = 'xxxxx'
    # file_prefix   = 'tmp_'
    # save_dir  = 'xxxxx'
    # scan_type = 'radial'
    
    if scan_type == 'radial':
        mag_factor = 38.0
        gamma = 0.65
        
    elif scan_type == 'carotid':
        mag_factor = 12.0
        gamma = 0.8

    
    flag_setpoint = False
    flag_heat = True
    flag_save_res  = True
    flag_save_proc_time = True
    flag_save_mid_res = False
    
    save_idx = 0
    save_imd_idx = 0
    
    if scan_type == 'radial':
        mag_factor = 38.0
        gamma = 0.65
        
    elif scan_type == 'carotid':
        mag_factor = 20.0 
        gamma = 0.80
    
    # start magnify videos
    aio = AccMagIO()
    print('- Loading data ...')
    stdata, _ = aio.load_pickle(file_dir=filepath_dir, file_prefix=file_prefix, 
                                time_clip=None, flag_setpoint=flag_setpoint) 
    print('Done! shape of the data: ', stdata.shape)
    
    accmag = AccMagCore(
        fps=30.0,
        freq_des=1.55,
        freq_range=[0.90, 1.9],
        butter_order=3.0,
        gamma=gamma 
    )
    
    # capturing images and start magnification
    elapsed_list = list()
    pbar = tqdm(stdata)
    for img in pbar:
        if flag_save_mid_res:
            mag_img, elapsed_t, acc_img, amp_img = accmag.magnify_log(
                img=img, 
                img_scale=None,
                mag_factor=mag_factor,
                flag_vis=False,
                flag_fast_vis=False,
            )
        else:
            mag_img, elapsed_t = accmag.magnify(
                img=img, 
                img_scale=None,
                mag_factor=mag_factor, 
                flag_vis=False,
                flag_fast_vis=False,
            )
        elapsed_list.append(elapsed_t)
        if flag_heat: # show heat map
            heatmapshow = cv2.applyColorMap(mag_img, 
                                            cv2.COLORMAP_JET)
            # cv2.imshow("Heatmap",  heatmapshow)
            # cv2.waitKey(1)
        
        # save result
        if flag_save_res:
            img     = np.stack((img, img, img), axis=2)
            mag_img = np.stack((mag_img, mag_img, mag_img), axis=2)
            res_img = np.hstack((img, mag_img, heatmapshow))
            # cv2.imshow("result image",  res_img)
            
            overlap = cv2.addWeighted(img, 0.5, heatmapshow, 0.5, gamma=0.5) 
            # cv2.imshow("overlap image",  overlap)
            
            cv2.imwrite(save_dir + '/mag/'  + str(save_idx) + '.jpg', mag_img)
            cv2.imwrite(save_dir + '/heat/' + str(save_idx) + '.jpg', heatmapshow)
            cv2.imwrite(save_dir + '/all/'  + str(save_idx) + '.jpg', res_img)
            cv2.imwrite(save_dir + '/overlap/'  + str(save_idx) + '.jpg', overlap)
            
            save_idx += 1
            # cv2.waitKey(1)
            
        if flag_save_mid_res:
            acc_path = save_dir + 'acc/'  + str(save_imd_idx) + '.pickle'
            amp_path = save_dir + '/amp/' + str(save_imd_idx) + '.pickle'
            pickle.dump(acc_img, open(acc_path, 'wb'))
            pickle.dump(amp_img, open(amp_path, 'wb'))
            save_imd_idx += 1
            
            
    if flag_save_proc_time:
        pickle.dump(
            elapsed_list, 
            open(save_dir+'time_ellapse.pickle', 'wb')
        )
        
    cv2.destroyAllWindows()
    
    plt.figure()
    plt.plot(np.array(elapsed_list)[1:]*1e+3)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # check_gpu()
    main()
    

