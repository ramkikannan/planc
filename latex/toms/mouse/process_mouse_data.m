% data from https://stanford.app.box.com/s/hvv2v6omrjwwa0nkanpdh67jt429wzq3
% frame indices: https://stanford.app.box.com/s/hvv2v6omrjwwa0nkanpdh67jt429wzq3/file/352459116715
% dataset: https://stanford.app.box.com/s/hvv2v6omrjwwa0nkanpdh67jt429wzq3/file/345234627309

% clear workspace
clear
fprintf('Trying to process mouse brain data file, reshaping and trimming frames across trials...\n');

% read frame indices of each trial
% - rows correspond to trial
% - 1st col: start of trial
% - 2nd col: light stimulus
% - 3rd col: water reward
% - 4th col: end of trial
% load frame info file (download if necessary)
if ~isfile('20161028_f4_try1.mat')
%if ~exist('20161028_f4_try1.mat', 'file')
    fprintf('Need to download frame info file from the following URL...\n https://stanford.app.box.com/s/hvv2v6omrjwwa0nkanpdh67jt429wzq3/file/352459116715\n');
    return
end
load 20161028_f4_try1
F = frame_indices; % var name is frame_indices
num_trials = size(F,1);

fprintf('Determining stimuli frames from frame info file...\n');

% check frames from start to light consistent
fs = F(:,2) - F(:,1);
assert(min(fs) == max(fs));
light_frame = min(fs)

% check frames from light to water consistent
fs = F(:,3) - F(:,2);
assert(min(fs) == max(fs));
water_frame = min(fs)

% reset end frame to be minimum frames from water to end
fs = F(:,4) - F(:,3);
F(:,4) = F(:,3) + min(fs);

% set common total number of frames
num_frames = F(1,4)-F(1,1)+1

% load data file (download if necessary)
if ~isfile('20161028_f4_try1.hdf5')
%if ~exist(20161028_f4_try1.hdf5','file')
    fprintf('Need to download data file from the following URL...\n https://stanford.app.box.com/s/hvv2v6omrjwwa0nkanpdh67jt429wzq3/file/345234627309\n');
    return;
end
fprintf('Reading original data...\n');
M = h5read('20161028_f4_try1.hdf5', '/Data/Images');
[width,height,total_frames] = size(M);

% copy trimmed data into new 4D array
%fprintf('Initializing trimmed array...\n');
%T = zeros(width,height,num_frames,num_trials,'uint16');
%for i = 1:num_trials
%    fprintf('Copying trial %d...\n',i);
%    T(:,:,:,i) = M(:,:,F(i,1):F(i,4));
%end
%
%fprintf('Saving trimmed 4D data to file...\n');
%save('20161028_f4_try0_trimmed_4D.mat','T','-v7.3');

% copy trimmed data into new 3D array
fprintf('Initializing trimmed array...\n');
T = zeros(width*height,num_frames,num_trials,'uint16');
for i = 1:num_trials
    fprintf('Copying trial %d...\n',i);
    T(:,:,i) = reshape(M(:,:,F(i,1):F(i,4)),[width*height,num_frames]);
end

fprintf('Saving trimmed 3D data to file...\n');
save('20161028_f4_try0_trimmed_3D.mat','T','height','width','light_frame','water_frame','num_frames','num_trials','-v7.3');
