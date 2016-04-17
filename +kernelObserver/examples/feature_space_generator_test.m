%===================== feature_space_generator_test =======================
%  
%  This code tests the FeatureSpaceGenerator class. 
%
%===================== feature_space_generator_test =======================
%
%  Name:	feature_space_generator_test.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/03/13
%  Modified: 2016/03/13
%
%===================== feature_space_generator_test =======================
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


%% load time-series data
generator = 'RBFNetwork';
k_type = 'gaussian';
scheme = 'smooth2'; % smoothly varying system 
load_file = ['./data/synthetic_time_series_generator_' generator ...
             '_kernel_' k_type '_scheme_' scheme '.mat'];       
load(load_file)

%% first, create finite-dimensional kernel model (RBFNetwork)
est_function_dim = 50;
basis = linspace(0, 2*pi, est_function_dim);
nbands = 5; 
nnoise = 5;
bandwidth_init = 0.6;
noise_init = 0.01; 
params_init = [bandwidth_init; noise_init];
param_optimizer = struct('method', 'likelihood', 'solver', 'primal',...
                         'Display', 'off');

%% now create FeatureSpaceGenerator object and infer parameters
fsg = kernelObserver.FeatureSpaceGenerator(basis, k_type, bandwidth_init, ...
                                           noise_init, param_optimizer);
tic
params = fsg.fit(orig_func_data, orig_func_obs);
t_end = toc; 
disp(['Training time for batch: ' num2str(t_end) ' seconds.'])


% visualize parameter stream
figure(1);
plot(params(1, :), 'b', 'LineWidth', 1.5)
xlabel('Time step')
ylabel('Bandwidth')
ylim([0.5, 1.5])
title('Bandwidth parameter over time')

figure(2);
plot(params(2, :), 'b', 'LineWidth', 1.5)
xlabel('Time step')
ylabel('Noise')
title('Noise parameter over time')


