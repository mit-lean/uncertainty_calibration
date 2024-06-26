import numpy as np
import os
import scipy.stats # normal distribution
import math
import time # debugging
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm # progress bar

"""
# Description: 
#         - calculate calibration metrics (ECE, RMSE) for 1) delta-1 accuracy vs. confidence, 2) observed ppf vs. expected ppf, 3) squared error vs. variance 
#         - plot calibration curve 
"""
class CalibrationCurve:
    def __init__(self, uncertainty_type, directory, output_directory, frame_start = 0, frame_end = -1, delta=0.25, warm_start=0, num_bins=100, img_dim_x=224,img_dim_y=224, epsilon=1e-7):        
        # hyperparameters
        self.delta = delta # delta interval for accuracy 
        self.epsilon = epsilon # added to variance to ensure no 0 variance (becomes nan for confidence array)
        self.num_bins = num_bins # number of bins for calibration curves
        self.warm_start = warm_start # number of frames to ignore at the start for calibration
        # directory names
        self.directory = directory 
        self.output_directory = output_directory # for saving 
        # sequence-specific parameters 
        self.img_dim_x = img_dim_x # dim x of prediction, ground-truth, and variance arrays
        self.img_dim_y = img_dim_y # dim y of prediction, ground-truth, and variance arrays
        self.frame_start = frame_start
        if frame_end == -1: # whole sequence 
            self.num_frames = self.calculate_num_frames() # calculate number of frames in directory
            self.frame_end = self.num_frames-1 # frame numbering starts from 0
        else:
            self.num_frames = frame_end-frame_start # calculate number of frames in directory 
            self.frame_end = frame_end
        # uncertainty type 
        self.uncertainty_type = uncertainty_type
        # initialize metrics
        self.total_delta1_acc = 0 
        self.accuracy_delta1_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames))# accuracy based on delta1 metric
        self.confidence_delta1_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames)) # confidence based on delta1 metric
        # initialize calibration variables 
        self.mask = np.ones((self.img_dim_x,self.img_dim_y,self.num_frames), dtype=bool) # mask for 0 values of ground-truth
        self.ppf_bins = OrderedDict()
        self.expected_p = np.arange(num_bins+1.0)/num_bins
        # initalize containers for error, variance, depth
        self.abs_error_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames)) # absolute error
        self.squared_abs_error_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames)) # squared absolute error
        self.rel_error_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames)) # relative error
        self.var_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames)) # variance
        self.rel_var_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames)) # var/mean
        self.gt_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames)) # ground-truth depth
        self.pred_unmasked = np.zeros((self.img_dim_x,self.img_dim_y,self.num_frames)) # predicted depth    
    def calculate_metrics(self):
        """
        Description: Calculate, bin, and plot all calibration metrics
        """
        sum_delta_1 = 0 # sum of delta_1 accuracy averaged across pixels in the frame 
        for dir in tqdm(range(self.frame_start, self.frame_end)):
            if dir >= self.warm_start:
                pred_array, var_array, gt_array = self.load_arrays(dir)
                self.make_gt_mask(dir, gt_array)
                self.calculate_accuracy_delta1(dir, pred_array, gt_array)
                sum_delta_1 += np.sum(self.accuracy_delta1_unmasked[:,:,dir][self.mask[:,:,dir]])/np.sum(self.mask[:,:,dir])
                self.calculate_confidence_delta1(dir, pred_array, var_array)
                self.calculate_expected_actual_ppf(dir, pred_array, var_array, gt_array)
                self.calculate_absolute_error(dir, pred_array, gt_array)
                self.calculate_relative_error(dir, pred_array, gt_array)
                self.calculate_relative_variance(dir, pred_array, var_array)
        self.total_delta1_acc = sum_delta_1/self.num_frames # calculate avg delta 1 
        # calculate bins from pixel-wise metrics, masking gets handled in self.bin_metrics() function 
        print("Computing bins for confidence vs. accuracy for delta_1="+str(1+self.delta))
        confidence_accuracy_bins = self.bin_metrics(self.confidence_delta1_unmasked, self.accuracy_delta1_unmasked)
        print("Computing bins for variance vs. squared absolute error")
        var_sq_error_bins = self.bin_metrics(self.var_unmasked, self.squared_abs_error_unmasked)
        print("Avg delta 1 method 2:" + str(self.total_delta1_acc))
        return confidence_accuracy_bins, var_sq_error_bins, self.ppf_bins, self.total_delta1_acc
    def load_arrays(self, dir):
        """ load arrays from results directory """
        gt_array = np.squeeze(np.load(self.directory+"/"+str(dir)+"/gt.npy")) # groundtruth depth
        pred_array = np.squeeze(np.load(self.directory+"/"+str(dir)+"/pred.npy")) # predicted depth 
        if self.uncertainty_type != "epistemic_plus_aleatoric": # if uncertainty is epistemic or aleatoric
            var_array = np.squeeze(np.load(self.directory+"/"+str(dir)+"/" + self.uncertainty_type +"_uncertainty.npy"))
        else: # handle case where uncertainty is epistemic + aleatoric 
            var_array = np.squeeze(np.load(self.directory+"/"+str(dir)+"/epistemic_uncertainty.npy")) + np.squeeze(np.load(self.directory+"/"+str(dir)+"/aleatoric_uncertainty.npy"))
        self.pred_unmasked[:,:,dir] = pred_array
        self.gt_unmasked[:,:,dir] = gt_array
        self.var_unmasked[:,:,dir] = var_array
        return pred_array, var_array, gt_array
    def make_gt_mask(self, dir, gt_array):
        """ return mask for values in gt array that are 0 """ 
        valid_mask = gt_array>0 # need valid mask since missing values in ground truth
        self.mask[:,:,dir] = valid_mask
        return
    def calculate_num_frames(self):
        """ Calculate number of frames in directory """
        num_frames = 0
        for dir in sorted(list(map(int,os.listdir(self.directory)))): # convert to int numbers for sorted
            num_frames+=1
        return num_frames
    def calculate_accuracy_delta1(self, dir, pred_array, gt_array):
        """ Calculate accuracy based on delta interval """
        upper_bound = (1.0+self.delta)*gt_array
        lower_bound = (1.0-self.delta)*gt_array
        delta1_pixelwise = np.logical_and((pred_array >= lower_bound),(pred_array <= upper_bound)).astype(float)
        self.accuracy_delta1_unmasked[:,:,dir] = delta1_pixelwise 
        return
    def calculate_confidence_delta1(self, dir, pred_array, var_array):
        """ Calculate confidence based on delta interval """
        var_array = var_array + self.epsilon # ensures no 0 variance (causes nan confidence)
        pixel_norm_dist = scipy.stats.norm(pred_array,np.sqrt(var_array))
        confidence_array = pixel_norm_dist.cdf(pred_array*(1.0+self.delta))-pixel_norm_dist.cdf(pred_array*(1.0-self.delta))
        assert (np.amin(pred_array) >= 0), "Problem, predicting negative depths!"
        assert (np.amin(var_array) >= 0), "Problem, predicting negative variance!"
        assert (np.amin(confidence_array) >= 0),"Problem, predicting negative confidence!"
        self.confidence_delta1_unmasked[:,:,dir] = confidence_array
        return
    def calculate_expected_actual_ppf(self, dir, pred_array, var_array, gt_array):
        """
        Calculate expected percent point function and observed percent point function 
        """
        pred_array = pred_array[self.mask[:,:,dir]]
        var_array = var_array[self.mask[:,:,dir]]
        gt_array = gt_array[self.mask[:,:,dir]]
        for p in self.expected_p:
            # mask directly because ppf doesn't go through bin metrics 
            ppf = scipy.stats.norm.ppf(p, pred_array, np.sqrt(var_array))
            obs_p = np.average(gt_array < ppf)
            if dir == 0: # if first dir, initialize dict entry as list
                self.ppf_bins[p] = []
            self.ppf_bins[p].append(obs_p)
    def calculate_absolute_error(self, dir, pred_array, gt_array):
        """
        Calculate absolute error for a single frame
        """
        pred_array, var_array, gt_array = self.load_arrays(dir)
        abs_error = np.absolute(gt_array-pred_array)
        self.abs_error_unmasked[:,:,dir] = abs_error
        self.squared_abs_error_unmasked[:,:,dir] = abs_error**2
        return
    def calculate_relative_error(self, dir, pred_array, gt_array):
        """
        Calculate relative error for a single frame
        """
        self.rel_error_unmasked[:,:,dir] = np.absolute((pred_array-gt_array)/gt_array) # may throw an error for divide by 0, later masked in binning
        return
    def calculate_relative_variance(self, dir, pred_array, var_array):
        """
        Calculate relative variance for a single frame
        """
        pred_array, var_array, gt_array = self.load_arrays(dir)
        self.rel_var_unmasked[:,:,dir] = var_array/(pred_array.astype(float)) 
        return

    def visualize_binned_pixels(self,x,y):
        """ 
        Visualize pixels in each image that are binned in each bucket (currently, 10 bins, 0<= x <= 0.1, 0.1 < x <= 0.2,...0.9 < x < = 1.0
        """
        # divide x into bins
        bins = OrderedDict() # initialize bin
        ## find max/min of x data
        x_max = np.amax(x)
        x_min = np.amin(x)
        x_bin_interval = (x_max-x_min)/10.
        for k in range(self.num_frames):
            # save the pixels that are have higher accuracy than confidence
            for i in range(1,11):
                # print("bin number: " + str(i))
                # sort which y's fall into which x bins
                if (i == 1): # include 0 lower limit in bucket
                    x_in_bin = np.logical_and(x[:,:,k] >= (i-1)*x_bin_interval+x_min, np.logical_and(x[:,:,k] <= i*x_bin_interval+x_min, y[:,:,k] > x[:,:,k]))
                else: # don't include lower limit in interval
                    x_in_bin = np.logical_and(x[:,:,k] > (i-1)*x_bin_interval+x_min, np.logical_and(x[:,:,k] <= i*x_bin_interval+x_min, y[:,:,k] > x[:,:,k]))
                x_in_bin = x_in_bin.astype(int)
                x_in_bin[np.where(self.mask[:,:,k] == False)[0], np.where(self.mask[:,:,k] == False)[1]] = 0 # masking here
                # print("Number of nonzero elements in x_in_bin: " + str(np.count_nonzero(x_in_bin)))
                rescaled = (255.0*x_in_bin).astype(np.uint8)
                # print("plotting and saving!")
                plt.imshow(rescaled)
                # store image
                if not os.path.exists(self.output_directory+"/debugging/"+str(k)+"/"):
                    os.makedirs(self.output_directory+"/debugging/"+str(k)+"/")
                    print(self.output_directory+"/debugging/"+str(k)+"/")
                plt.savefig(self.output_directory+"/debugging/"+str(k)+"/"+str(i)+".png")
                plt.close()
    def bin_metrics(self, x, y):
        """ 
        Bin y based on intervals of x 
        """
        # mask x and y
        # print("Shape of x, y before masking: " + str(x.shape) + ", " + str(y.shape))
        x = x[self.mask]
        y = y[self.mask]
        # print("Shape of x, y after masking: " + str(x.shape) + ", " + str(y.shape))
        # divide x into bins
        bins = OrderedDict() # initialize bin
        ## find max/min of x data
        x_max = np.amax(x)
        x_min = np.amin(x)
        x_bin_interval = (x_max-x_min)/float(self.num_bins)
        if x_max > 1e6:
            print("NOTE: X max is large. X_max: " + str(x_max) + ", x_bin_interval: " + str(x_bin_interval))
        elif x_min < 0:
            print("NOTE: X min is negative.")
            if x_min < -1e-6:
                print("NOTE: X min is negative and small. X_min: " + str(x_min) + ", x_bin_interval: " + str(x_bin_interval))
        # sort which y's fall into which x bins
        for i in tqdm(range(1,int(self.num_bins)+1)):
            if (i == 1): # include 0 lower limit in bucket
                x_in_bin = np.logical_and(x >= (i-1)*x_bin_interval+x_min, x <= i*x_bin_interval+x_min)
            else: # don't include lower limit in interval
                x_in_bin = np.logical_and(x > (i-1)*x_bin_interval+x_min, x <= i*x_bin_interval+x_min)
            # if at least one value in the bin, and bin interval key not yet in bin dict
            mid_interval = i*x_bin_interval+x_min-x_bin_interval/2 
            if mid_interval not in bins and np.any(x_in_bin):
                # bin key is middle of interval 
                bins[mid_interval] = np.ndarray.flatten(y[x_in_bin])
            # if at least one value in the bin
            elif np.any(x_in_bin):
                # bin key is middle of interval 
                bins[mid_interval] = np.hstack((bins[mid_interval],np.ndarray.flatten(y[x_in_bin])))
        return bins
    def calculate_ece(self,bins):
        """ 
        Calculate expected calibration error (ECE)
        """
        # calculate all values
        all_values_counter = 0
        for key, value in sorted(bins.items()):
            if isinstance(value, list):
                all_values_counter += len(value)
            else:
                all_values_counter += value.shape[0]
        # calculate ECE
        ECE = 0
        for key, value in bins.items():
            ECE = ECE + (float(len(value))/all_values_counter)*abs(key - np.average(value))
        return ECE
    def calculate_rmse(self,bins): 
        """ 
        Calculate root mean squared error (RMSE)
        """ 
        RMSE = 0
        # calculate RMSE
        for key, value in bins.items():
            RMSE  = RMSE + (np.average(value)-key)**2
        RMSE = math.sqrt(RMSE/len(bins))
        return RMSE
