'''
Author: Dianye Huang
Date: 2023-02-16 14:29:54
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2023-07-04 17:34:02
Description: 
    Utils for the video aceleration magnification algorithm.
    
Notations:
    - H: height resolution of an image
    - W: width  resolution of an image 
    - C: number of image channels
    - N: number of frames in a spatio-temporal dataset
    - L: number of scales of an image pyramid
'''

import os
import cv2
import pickle
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

import torch

class AccMagIO:
    '''
        This class implements an interface for capturing 
        a US image for magnification, and also loading
        the demo data
    '''
    def __init__(self):
        pass
        
    def load_pickle(self, file_dir, file_prefix, 
                time_clip=None, flag_setpoint=False):
        '''
        Description:
            Loading pickle data to get a stack of ultrasound images
        @ Input:
            - file_dir       (str): pickle file's directory
            - file_prefix    (str): pickle file's name without '.pickle' 
            - time_clip     (list): time index, from {time_clip[0]} second to {time_clip[0]} second
            - flag_setpoint (bool): True -> setpoint video, False -> continuous scan video
        @ Output:
            - stdata (np.array): spatio-temporal data (N, H, W)
        '''
        
        if flag_setpoint:
            filepath = file_dir+file_prefix+'.pickle'
            stdata = np.array(pickle.load(open(filepath, 'rb'))['us_images'])
            num_frame = len(stdata)
            
        else:
            if time_clip is None:
                time_clip = [int(0),len(os.listdir(file_dir))]   
            
            num_frame = 0
            stdata = list()
            for i in range(time_clip[0], time_clip[1]):    # load time period
                filepath = file_dir+file_prefix+str(i)+'.pickle'
                if not os.path.exists(filepath):
                    break
                tmp_data = np.array(pickle.load(open(filepath, 'rb'))['us_images'])
                stdata.extend(tmp_data)
                num_frame += 1
            stdata = np.array(stdata)
        return stdata, num_frame
    
    
from scipy import signal
class AccMagFilter:
    '''
        This class implements realtime filters
    '''
    def __init__(self):
        
        self.butter_xbuffer = None # input  states of the butter worth filter
        self.butter_ybuffer = None # output states
        
        self.dog_buffer = None     # buffer for the DoG filter
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_butter_bp_params(self, freq_range=[0.9, 2.5], fps=30.0, order=4):
        '''
        Description:
            Get butterworth bandpass filter parameters with given pass band 
            and sampling rate.
        @  Input:
            - freq_range (list): low and high cutoff frequencies.
            - fps       (float): frame per second
            - order     (float): order of the butterworth filter
        @ Output:
            - (b, a) (a tuple of two np.arrays):   
                        b is the nominator of the discrete transfer
                        function, and a denotes the denominators
        '''
        return signal.butter(order, [2*freq_range[0]/fps, 
                            2*freq_range[1]/fps], btype='bandpass')
    
    def init_butter_bp_buffer(self, b, a, input_shape):
        '''
        Description:
            Initialize a global buffer for bandpass filtering
        @  Input: 
            - b       (np.array): nominator coeffs of the butter worth transfer function
            - a       (np.array): denominator coeffs 
            - input_shape (list): size of the image, the buffer size is of (h, w)
        '''
        # input variables  -> X 
        self.butter_xbuffer = np.zeros(
                        (len(b), input_shape[0], input_shape[1])
                    )  
        self.butter_xbuffer = torch.tensor(
            self.butter_xbuffer, requires_grad=False
        ).to(self.device)
        
        # output variables -> Y
        self.butter_ybuffer = np.zeros(
                        (len(a), input_shape[0], input_shape[1])
                    )  
        self.butter_ybuffer = torch.tensor(
            self.butter_ybuffer, requires_grad=False
        ).to(self.device)
        

    def roll_and_append(self, previous, current):
        '''
        Description:
            Rolling a sliding buffer and insert an input element  
        @  Input: 
            - previous (np.array): a sliding buffer of size (N, H, W)
            - current  (np.array): an input image of size (H, W)
        @ Output:
            - a sliding and concatenated new array
        '''
        return torch.cat((previous[1:], current.unsqueeze(0)), dim=0) # .to(self.device)

    def update_bp_filter(self, input, rn_b, rn_a):
        '''
        Descriotion:
            Receive input image X and calc output Y with the buffer data
            Then update the buffer with X and Y images
        @  Input: 
            - input    (np.array):  input image of size (H, W)
            - rn_b     (np.array):  reversed normalized nominator of the transfer function
                                    normalized means being divided by a[0]
                                    reversed means rn_b = n_b.reverse()
                                    size is (len(b), 1, 1)
            - rn_a     (np.array):  normalized denominator of the transfer function
                                    size is (len(a)-1, 1, 1), a[0] is wiped out
        @ Output:
            - yn       (np.array):  output of the butterworth filter, size -> (H, W)
        '''
        # update input x_buffer, sliding and concatenate
        self.butter_xbuffer = self.roll_and_append(
                                previous=self.butter_xbuffer,
                                current=input
                            )
        
        # get new output 
        yn = torch.sum(self.butter_xbuffer*rn_b, dim=0) - \
                    torch.sum(self.butter_ybuffer[1:]*rn_a, dim=0)
                    
        # update output y_buffer, sliding and concatenate
        self.butter_ybuffer = self.roll_and_append(
                                previous=self.butter_ybuffer,
                                current=yn   # insert the newest output to the buffer
                            )
        return yn
    
    def get_DoG_1d_kernel(self, freq_des=1.5, fps=30.0, flag_vis=False):
        """
        @ Description: 
            Difference of Gaussian (DoG) operator is an approximation of 
            the Laplacian of Gaussian (LoG) operator. This function build
            a DoG operator according to the given desired frequency. DoG
            should meet two requirements:
                -  i). sum(DOG_kernel) = 0
                - ii). sum(abs(DOG_kernel)) = 1
        @ Input:
            - freq_des (float): desired frequency
            - fps      (float): frame per second
            - flag_vis (bool) : decide whether to plot the generated kernel
        @ Output: 
            - DoG_kernel (np.array): a three dimensional vector kernel 
        """
        t_inv      = 1/(4*freq_des) 
        frame_inv  = np.ceil(fps*t_inv).astype(int)
        windowSize = 2*frame_inv
        signalLen  = windowSize                   
        sigma      = frame_inv/2
        x          = np.linspace(-signalLen/2, signalLen/2, signalLen+1)
        
        sigma1 = sigma/2
        sigma2 = sigma*2
        
        gaussFilter1 = np.exp(-np.power(x, 2) / (2 * sigma1**2))
        gaussFilter1 = gaussFilter1 / sum (gaussFilter1)
        gaussFilter2 = np.exp(-np.power(x, 2) / (2 * sigma2**2))
        gaussFilter2 = gaussFilter2 / sum (gaussFilter2)
        DoG_kernel   = gaussFilter1-gaussFilter2
        
        # normalization so that maximum value of the kernel is 1.
        DoG_kernel = DoG_kernel/sum(abs(DoG_kernel)); # 'Normalization'
        
        if flag_vis:
            print('frame interval: ', frame_inv)
            print('window size: ', x.shape[0])
            index = [i for i in range(len(DoG_kernel))]
            plt.figure()
            plt.plot(DoG_kernel)
            plt.scatter(index, DoG_kernel)
            plt.grid(True)
            plt.show()

        return torch.tensor(
            DoG_kernel[:, None, None], requires_grad=False
        ).to(self.device)
    
    def init_DoG_buffer(self, kernel_len, input_shape):
        self.dog_buffer = np.zeros((kernel_len, input_shape[0], 
                                                input_shape[1]))
        self.dog_buffer = torch.tensor(
            self.dog_buffer, requires_grad=False
        ).to(self.device)
    
    def update_DoG_filter(self, kernel, img):
        self.dog_buffer = self.roll_and_append(
            previous=self.dog_buffer,
            current=img
        )
        return torch.sum(self.dog_buffer*kernel, dim=0)
    
    def ideal_temporal_bp(self, stdata_arr, fps, freq_range, 
                    mag_factor=1.0, axis=0, package_use='numpy'):
        '''
        Ideal temporal band pass filter
        @ Input : 
            - stdata_arr(np.array): spatio-temporal image data whose shape is (N, H, W)
            - fps          (float): frame per second
            - freq_range    (list): [low, heigh], frequency band
        @ Output: 
            - result    (np.array): filtered result
        '''
        if package_use == 'scipy':
            fft = fftpack.rfft(stdata_arr, axis=axis)
            frequencies = fftpack.fftfreq(stdata_arr.shape[0], d=1.0 / fps)
            bound_low = (np.abs(frequencies - freq_range[0])).argmin()
            bound_high = (np.abs(frequencies - freq_range[1])).argmin()
            fft[:bound_low] = 0
            fft[bound_high:-bound_high] = 0
            fft[-bound_low:] = 0
            result = np.ndarray(shape=stdata_arr.shape, dtype='float')
            result[:] = mag_factor*fftpack.ifft(fft, axis=0).real
        elif package_use == 'numpy':
            fft = np.fft.fft(stdata_arr, axis=axis)
            frequencies = np.fft.fftfreq(stdata_arr.shape[0], d=1.0/fps)
            low = (np.abs(frequencies - freq_range[0])).argmin()
            high = (np.abs(frequencies - freq_range[1])).argmin()
            fft[:low] = 0
            fft[high:] = 0
            result = mag_factor*np.fft.ifft(fft, axis=0).real 
        return result
    
class AccMagPyramid:
    def __init__(self, buff_size = 11, twin_size=60, pyr_type='Gaussian', pyr_level=3):
        
        self.pyr_type  = pyr_type
        self.pyr_level = int(pyr_level)
        
        self._pyr_buff = None
        self._pyr_acc_buff = None
        self.buff_size = buff_size
        self.twin_size = twin_size

    def get_image_gaussian_pyramid(self, image, pyramid_level=None):
        '''
        Description:
            Create a gaussian pyramid of a single image
        @ Input : 
            - image   (np.array): a single image with shape = (H, W)
        @ Output: 
            - img_pyramid (list): pyramid list with shape = (L, Hv, Wv) where v denotes varying sizes
        '''
        if pyramid_level is None:
            pyramid_level = self.pyr_level
            
        gauss_copy = np.ndarray(shape=image.shape, dtype="float")
        gauss_copy[:] = image        
        img_pyramid = [gauss_copy]
        
        for _ in range(1, pyramid_level):
            gauss_copy = cv2.pyrDown(gauss_copy)
            img_pyramid.append(gauss_copy)
            
        return img_pyramid
    
    def get_image_laplacian_pyramid(self, image, pyramid_level=None):
        '''
        Description:
            Create a laplacian pyramid of a single image
        @ Input : 
            - image   (np.array): a single image with shape = (H, W)
        @ Output: 
            - laplacian_pyramid (list): pyramid list with shape = (L, Hv, Wv) where v denotes varying sizes
        '''
        if pyramid_level is None:
            pyramid_level = self.pyr_level
            
        gauss_pyramid = self.get_image_gaussian_pyramid(image, pyramid_level)
        laplacian_pyramid = []
        
        for i in range(pyramid_level - 1):
            laplacian_pyramid.append((gauss_pyramid[i] - cv2.pyrUp(gauss_pyramid[i + 1])) + 0)
        laplacian_pyramid.append(gauss_pyramid[-1])
        
        return laplacian_pyramid
    
    def get_image_raw_pyramid(self, image, pyramid_level=None):
        '''
        Description:
            Create a raw pyramid of a single image by direct downsampling operation
        @ Input : 
            - image   (np.array): a single image with shape = (H, W)
        @ Output: 
            - pyramid     (list): pyramid list with shape = (L, Hv, Wv) where v denotes varying sizes
        '''
        if pyramid_level is None:
            pyramid_level = self.pyr_level
            
        pyramid = [image.copy()]
        gauss_copy = np.ndarray(shape=image.shape, dtype="float")
        gauss_copy[:] = image 
        
        for _ in range(1, pyramid_level):
            image = image[0:image.shape[0]:2, 0:image.shape[1]:2]
            pyramid.append(image)
        return pyramid
    
    def get_image_pyramid(self, image, pyramid_level=None, pyramid_type=None):
        '''
        Description:
            Create an image pyramid of pyramid_type at pyramid_level scales
        @ Input : 
            - image   (np.array): a single image with shape = (H, W)
            - pyramid_level (float): number of scales
            - pyramid_type    (str): type of the pyramid
        @ Output: 
            - pyramid     (list): pyramid list with shape = [L],(Hv, Wv) where v denotes varying sizes
        '''
        if pyramid_level is None:
            pyramid_level = self.pyr_level
            
        if pyramid_type is None:
            pyramid_type = self.pyr_type
            
        if   self.pyr_type == 'Gaussian':
            img_pyr = self.get_image_gaussian_pyramid(image=image, 
                                        pyramid_level=pyramid_level)
        elif self.pyr_type == 'None':
            img_pyr = self.get_image_raw_pyramid(image=image, 
                                        pyramid_level=pyramid_level)
        elif self.pyr_type == 'Laplacian':
            img_pyr = self.get_image_laplacian_pyramid(image=image, 
                                        pyramid_level=pyramid_level)
            
        return img_pyr
    
    def recon_magimg_pyramid(self, img_pyr, pyramid_level=None):
        if pyramid_level is None:
            pyramid_level = self.pyr_level
            
        mag_img = img_pyr[-1]
        for level in range(pyramid_level):
            if level != pyramid_level - 1:
                mag_img = cv2.pyrUp(mag_img)
                mag_img += img_pyr[pyramid_level-level-2]
                
        return mag_img
    
    def roll_and_append(self, previous, current):
        return np.vstack((previous[1:, :, :], current[None, :, :]))
    
    def update_pyrs_buffer(self, pyr_new):
        '''
            Update pyramid in a sliding manner, the sliding window is self.buff_size
        '''
        # initialize buffer
        if self._pyr_buff is None:
            self._pyr_buff = list()
            
            for pyr in pyr_new:
                bsize = pyr.shape
                self._pyr_buff.append(np.zeros((self.buff_size, 
                            bsize[0], bsize[1]), dtype=np.float32))
                
        # update buffer
        for level, img in enumerate(pyr_new):
            self._pyr_buff[level] = self.roll_and_append(
                                        previous=self._pyr_buff[level],
                                        current=img
                                    )
            
    def get_pyrs_buffer(self):
        return self._pyr_buff
    
    def get_pyr_from_buffer(self, idx):
        pyr = list()
        for tmp in self._pyr_buff:
            pyr.append(tmp[idx])
        return pyr
        

if __name__ == '__main__':
    # Test of loading data
    # filepath_dir = 'xxxx' 
    # file_prefix = 'tmp_'
    # aio = AccMagIO()
    # print('- Loading data ...')
    # stdata = aio.load_pickle(file_dir=filepath_dir, file_prefix=file_prefix, 
    #                         time_clip=[int(3), int(10)], flag_setpoint=False)
    # print('Done! shape of the data: ', stdata.shape)
    # for img in stdata:
    #     cv2.imshow('raw image', img)
    #     cv2.waitKey(33)
    # cv2.destroyAllWindows()
    
    # Test of getting butter worth filter params
    afilter = AccMagFilter()
    b, a = afilter.get_butter_bp_params(
        fps=30.0,
        freq_range=[0.9, 2.5],
        order=3
    )
    