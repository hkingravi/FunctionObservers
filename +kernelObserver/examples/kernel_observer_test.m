%======================== kernel_observer_test ============================
%  
%  This code tests the FeatureSpaceGenerator class. 
%
%======================== kernel_observer_test ============================
%
%  Name:	kernel_observer_test.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/04/21
%  Modified: 2016/04/21
%
%======================== kernel_observer_test ============================
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

font_size = 15;
line_width = 2.5;

%% load time-series data
generator = 'RBFNetwork';
k_type = 'gaussian';
scheme = 'smooth2'; % smoothly varying system 
load_file = ['./data/synthetic_time_series_generator_' generator ...
             '_kernel_' k_type '_scheme_' scheme '.mat'];       
load(load_file)

nsamp_ret = 50;  % truncate series for now
orig_func_data = orig_func_data(:, :, 1:nsamp_ret);
orig_func_obs = orig_func_obs(:, :, 1:nsamp_ret);

%% first, create finite-dimensional kernel model (RandomKitchenSinks)
nbases = 300;
ndim = 1; 
k_type = 'gaussian';
bandwidth = 1; 
noise = 0.01;
optimizer = struct('method', 'likelihood', 'solver', 'primal', ...
                   'Display', 'off', 'DerivativeCheck', 'off');
rks = kernelObserver.RandomKitchenSinks(nbases, ndim, k_type, ...
                                        bandwidth, noise,...
                                        optimizer);
meas_type = 'random';
nmeas = 100;

%% now create KernelObserver object and infer parameters
kobs = kernelObserver.KernelObserver(rks, nmeas, meas_type);
tic
kobs.fit(orig_func_data, orig_func_obs);
param_stream = kobs.get_param_stream();
t_end = toc; 
disp(['Training time for batch: ' num2str(t_end) ' seconds.'])

K = kobs.get('K');

% visualize parameter stream
figure(1);
plot(param_stream(1, :), 'b', 'LineWidth', line_width)
xlabel('Time step')
ylabel('Bandwidth')
title('Bandwidth parameter over time')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(2);
plot(param_stream(2, :), 'b', 'LineWidth', line_width)
xlabel('Time step')
ylabel('Noise')
title('Noise parameter over time')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(3);
imagesc(K')
t_rks = ['Measurement operator (' meas_type ') for Kernel Observer (RKS) with'...
         num2str(nmeas) ' measurements'];
title(t_rks)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

set(figure(1),'Position',[100 100 800 600]);
set(figure(2),'Position',[100 100 800 600]);
set(figure(3),'Position',[100 100 800 600]);
