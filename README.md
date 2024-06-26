# Uncertainty Calibration Metrics
This repo implements uncertainty calibration metrics for depth estimation. The following calibration metrics are currently supported:
1. Expected calibration error (ECE) and root mean squared error (RMSE) for confidence within delta interval vs. accuracy within delta interval from ground-truth 
2. ECE and RMSE for variance vs. squared error 
3. ECE and RMSE for expected probability point function (PPF) vs. observed PPF

# Dependencies 
This requires NumPy, matplotlib, tqdm, SciPy libraries and is tested with Python 3.8. To install the dependencies, create and activate a new conda environment with Python 3.8, and run `installation.sh` 
<!-- numpy 1.19.2
matplotlib 3.3.4
tqdm 4.62.3
scipy 1.5.2 -->
# Running calibration 
To run a calibration example, run the following:
```
python3 examples/calibrate.py <UNCERTAINTY_TYPE> <PATH_TO_DEPTH_RESULTS_DIR> <PATH_TO_CALIBRATION_RESULTS_DIR> 
```
where `UNCERTAINTY_TYPE` is the name of the uncertainty array stored that you want to calibrate (e.g., `epistemic`, `aleatoric`, or `epistemic_plus_aleatoric`). In addition, within `examples/calibrate.py`, the following parameters with their default values can also be set:
- `delta_interval = 0.25`: (1+delta = delta_x metric, set equal to 0.25 for original delta_1 metric which is accurate within 25% of ground-truth)
- `frame_start = 0`: starting frame idx 
- `frame_end = -1`: either ending frame idx or if set to -1, then full sequence 
- `warm_start = 0`: number of frames to ignore at the start of the calibration (set equal to 0 to analyze all frames)

Currently, the expected contents of the DNN results directory is as follows: 
* <PATH_TO_DEPTH_RESULTS_DIR>
    * 0 # frame number in sequence
        * gt.npy # ground-truth depth
        * pred.npy # predicted depth 
        * aleatoric_uncertainty.npy # aleatoric uncertainty 
        * epistemic_uncertainty.npy # epistemic uncertainty 
        * otherUncertaintyType_uncertainty.npy # can handle other uncertainty names, e.g., multiview_uncertainty.npy
    * 1 
        * ...