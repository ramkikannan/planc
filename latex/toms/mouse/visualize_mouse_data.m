% script to visualize mouse data
load 20161028_f4_try0_trimmed_3D.mat

T = reshape(T,[width,height,num_frames,num_trials]);

trial = 13;

handle = implay(T(:,:,:,trial));
handle.Visual.ColorMap.UserRange = 1; 
handle.Visual.ColorMap.UserRangeMin = min(T(:)); 
handle.Visual.ColorMap.UserRangeMax = max(T(:));