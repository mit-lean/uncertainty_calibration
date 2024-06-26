import numpy as np
import matplotlib.pyplot as plt 
import os 

def plot_bins(bins, output_directory, delta_interval, title, x_axis_label, y_axis_label, x_save, y_save, ERROR_BARS=False, BIN_COUNT=True, SAVE_FIG=True):
        """ 
        Plot and save calibration metrics (with slope = 1 perfect calibration line)
        """
        x = []
        y = []
        bin_count = []
        small_sample_markers_x = []
        small_sample_markers_y = []
        all_values_counter = 0
        if ERROR_BARS:
            y_std_dev = []
        for key, value in sorted(bins.items()):
            x.append(key)
            y.append(np.average(value))
            if isinstance(value, list):
                if (len(value)<10):
                    small_sample_markers_x.append(key)
                    small_sample_markers_y.append(np.average(value))
                all_values_counter += len(value)
                bin_count.append(len(value))
            else:
                if (value.shape[0]<10):
                    small_sample_markers_x.append(key)
                    small_sample_markers_y.append(np.average(value))
                all_values_counter += value.shape[0]
                bin_count.append(value.shape[0])
            if ERROR_BARS:
                y_std_dev.append(np.std(value))
        if BIN_COUNT:
            fig, (ax1, ax2) = plt.subplots(1, 2)
        else:
            fig, (ax1) = plt.subplots(1, 1)
        fig.suptitle(title)
        if ERROR_BARS:
            ax1.fill_between(np.asarray(x), (np.asarray(y) - np.asarray(y_std_dev)), np.asarray(y) + np.asarray(y_std_dev), color='lightblue',interpolate=True)
        ax1.scatter(x, y)
        ax1.scatter(small_sample_markers_x, small_sample_markers_y, marker='*',color='black',s=75)
        ax1.set_xlabel(x_axis_label,fontsize=14)
        ax1.set_ylabel(y_axis_label, fontsize=14)
        # plot straight line
        x_slope_1 = [min(bins.keys()),max(bins.keys())]
        y_slope_1 = [min(bins.keys()),max(bins.keys())]
        ax1.plot(x_slope_1,y_slope_1,'--',color='gray', label='perfect calibration')
        if BIN_COUNT: # plot number of pixels in each bin vs. x-axis
            ax2.scatter(x,bin_count)
            ax2.set_xlabel(x_axis_label,fontsize=14)
            ax2.set_ylabel("Number of pixels in bin", fontsize=14)
        plt.tight_layout() # adjust plot so not squashed 
        if SAVE_FIG:
            if not os.path.exists(output_directory+"/plots/"):
                os.makedirs(output_directory+"/plots/")
            plt.savefig(output_directory+"/plots/"+title+".pdf")
            plt.savefig(output_directory+"/plots/"+title+".png")
            # save numpy data to plot again
            if "Delta-1 " in y_axis_label: # include delta 1 in saved numpy array names 
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_"+str(delta_interval)+"_x.npy", x)
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_"+str(delta_interval)+"_y.npy", y)
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_"+str(delta_interval)+"_small_sample_markers_x.npy", small_sample_markers_x)
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_"+str(delta_interval)+"_small_sample_markers_y.npy", small_sample_markers_y)
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_"+str(delta_interval)+"_bin_count.npy", bin_count)
            else:
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_x.npy", x)
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_y.npy", y)
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_small_sample_markers_x.npy", small_sample_markers_x)
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_small_sample_markers_y.npy", small_sample_markers_y)
                np.save(output_directory+"/plots/"+x_save+"_"+y_save+"_bin_count.npy", bin_count)
        return fig

def plot_distribution_of_bins(bins):
    """ 
    Plot distribution of values within each bin
    """
    for key, value in sorted(bins.items()):
        plt.figure()
        plt.hist(value)
        plt.title("Bin value: " + str(key))