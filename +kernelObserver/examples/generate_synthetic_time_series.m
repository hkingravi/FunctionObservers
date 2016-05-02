%==================== generate_synthetic_time_series ======================
%  
%  This code generates a synthetic time series from a sequence of weights,
%  for use in testing other classes in this library. 
%
%==================== generate_synthetic_time_series ======================
%
%  Name:	generate_synthetic_time_series.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/03/13
%  Modified: 2016/03/13
%
%==================== generate_synthetic_time_series ======================
clc; clear all; clear classes; close all

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('./data')
addpath('./utils')

% set seed
seed = 20; 
s = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(s);

%% time-varying data arises from weight set associated to an RBF network
eval_data = 0:0.005:2*pi; % where the function values are evaluated for plotting purposes 
orig_function_dim = 10;
centers_orig = linspace(0, 2*pi, orig_function_dim); % we assume that these are shared 
generator = 'RBFNetwork';
scheme = 'smooth2'; % smoothly varying system 
sys_type = 'continuous'; 

% create rbf network to generate the time-varying function
k_type = 'gaussian';
bandwidth = [0.3, 0.1]; % pass in multiple bandwidths to check correct handling
noise = [0.1, 2]; % pass in multiple noise parameters to check correct handling
model = kernelObserver.RBFNetwork(centers_orig, k_type, bandwidth, noise); 

init_weights_orig = randn(orig_function_dim,1);
weights = init_weights_orig;
dt = 0.03; 
current_t = 0; 
final_time = 50;
num_steps = floor(final_time/dt); 

nsamp = 500;
nsamp_plot = length(eval_data); % number of samples for plotting
data = linspace(0, 2*pi, nsamp); % where the function values are sampled from

times = zeros(1, num_steps);
ideal_weight_trajectory = zeros(orig_function_dim, num_steps); 
orig_func_vals = zeros(1, nsamp, num_steps);
orig_func_plot_vals = zeros(1, nsamp_plot, num_steps);
orig_func_data = zeros(1, nsamp, num_steps);
orig_func_obs = zeros(1, nsamp, num_steps);

% compute times; used to generate time-varying function
for i=1:num_steps
  times(i) = current_t;
  current_t = dt*i;
end

for i=1:num_steps  
  weights = time_varying_uncertainty(weights, times(i), scheme);  % get weights
  
  model.set('weights', weights);
  orig_func = model.predict(data); 
  orig_func_plot = model.predict(eval_data); 
  orig_obs = orig_func + noise(1, 1)*randn(1, nsamp);
    
  % store function values for error plots later   
  orig_func_vals(:, :, i) = orig_func;  
  orig_func_plot_vals(:, :, i) = orig_func_plot;  
  orig_func_data(:, :, i) = data;  
  orig_func_obs(:, :, i) = orig_obs;  
end

% save observations to a file
save_file = ['./data/synthetic_time_series_generator_' generator ...
             '_kernel_' k_type '_scheme_' scheme '.mat'];
save(save_file, 'orig_func_data', 'orig_func_obs', 'orig_func_plot_vals')