%======================== kernel_observer_test ============================
%  
%  This code tests the KernelObserver class. 
%
%======================== kernel_observer_test ============================
%
%  Name:	kernel_observer_test.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/04/21
%  Modified: 2016/05/10
%
%======================== kernel_observer_test ============================
clc; clear; close all

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
save_results = 1; 
ext = 'png';

%% Time-Series Data Loading and Setup
generator = 'RBFNetwork';
k_type = 'gaussian';
scheme = 'smooth2'; % smoothly varying system 
load_file = ['./data/synthetic_time_series_generator_' generator ...
             '_kernel_' k_type '_scheme_' scheme '.mat'];       
load(load_file)

tic
nsteps = size(orig_func_data, 3);
orig_func_data_cell = cell(1, nsteps);
orig_func_obs_cell = cell(1, nsteps);
for i=1:nsteps  
  orig_func_data_cell{i} = orig_func_data(:, :, i) - ...
                           mean(orig_func_data(:, :, i));  % make data zero mean; important for dynamics
  orig_func_obs_cell{i} = orig_func_obs(:, :, i);
end
cell_time = toc;
disp(['Time taken to construct data cells: ' num2str(cell_time)])

nsamp_tr = 100;  % truncate series for now
nsamp_te_start = nsamp_tr + 20;  % predict beyond these values
func_data_tr = orig_func_data_cell(1:nsamp_tr);
func_obs_tr = orig_func_obs_cell(1:nsamp_tr);
func_data_te = orig_func_data_cell(nsamp_te_start:nsteps);
func_obs_te = orig_func_obs_cell(nsamp_te_start:nsteps);
meas_data = func_data_tr{1};  % we will subsample the data locations from this matrix

sorted = 0;
xmin = min(orig_func_data_cell{1});
xmax = max(orig_func_data_cell{1});
ymin = -36;
ymax = 36;
use_plot_min = 1;

%% first, create finite-dimensional kernel model (RandomKitchenSinks)
nbases = 300;
ndim = 1; 
k_type = 'gaussian';
bandwidth = 1; 
noise = 0.01;
optimizer = struct('method', 'likelihood', 'solver', 'primal', ...
                   'Display', 'off', 'DerivativeCheck', 'off', 'sort_mat', sorted);
rks = kernelObserver.RandomKitchenSinks(nbases, ndim, k_type, ...
                                        bandwidth, noise,...
                                        optimizer);
meas_type = 'random';
nmeas = 100;

%% create KernelObserver object and infer parameters
param_struct = struct();  % pass in empty struct to use defaults for filter
kobs = kernelObserver.KernelObserver(rks, nmeas, meas_type, param_struct);
tic
meas_inds = kobs.fit(func_data_tr, func_obs_tr, meas_data);
meas_actual = meas_data(:, meas_inds);
param_stream = kobs.get_param_stream();
train_time = toc; 
disp(['Time taken to train on ' num2str(nsamp_tr) ' time steps: '...
       num2str(train_time) ' seconds.'])

K = kobs.get('K');
filter = kobs.get('filter');
A = filter.get('A');

%% Functional Output Selection 
% select which type of output you're interested in seeing; default is the
% tracking error, while the other option is visualization of the functions
output = 'error'; % 'error' or 'function'

if strcmp(output,'function')
  vidObj_auto_vs_obs = VideoWriter(['./results/kobs_function_auto_vs_obs_' scheme ...
                                    '_rks_nmeas_' num2str(nmeas) '.avi'] );
  open(vidObj_auto_vs_obs);
end

%% Prediction Section
% utilize KernelObserver to predict output 
tic
nsamp_te = size(func_data_te, 2);
preds_te = cell(1, nsamp_te);
rms_error_te = zeros(1, nsamp_te);
te_times = zeros(1, nsamp_te);
weights_stream = zeros(2*nbases, nsamp_te);
for i=1:nsamp_te
  % make predictions on the entire dataset using current weights
  f = kobs.predict(func_data_te{i});
  rms_error_te(i) = norm(f - func_obs_te{i});
  te_times(i) = i;
  preds_te{i} = f;  
  weights_stream(:, i) = kobs.get('curr_weights');
  
  % utilize measurements from certain locations to correct observer state
  current_meas = func_obs_te{i}(meas_inds) + 0.1*randn(1, nmeas); 
  kobs.update(current_meas');  
  
  % create video, if asked for
  if strcmp(output, 'function')
    % create arrays for plotting
    pred_data = func_data_te{i}(meas_inds); 
    pred_obs = func_obs_te{i}(meas_inds); 
    eval_data = func_data_te{i};
    orig_func_plot = func_obs_te{i}; 
    pred_obs_plot = f; 
    
    figure(1);
%     subplot(1,2,1)        
    plot(eval_data, orig_func_plot, 'g', 'LineWidth', 3);
    hold on;
    plot(eval_data, pred_obs_plot, 'b--', 'LineWidth', 3);
    plot(pred_data, pred_obs, 'ro', 'MarkerSize', 10);
    hold off
    if use_plot_min == 1
      xlim([xmin xmax])
      ylim([ymin ymax])
    end
    set(gca, 'FontSize', 20)
    h_legend = legend('original', 'observer');
    set(h_legend, 'FontSize',20);
    
%     subplot(1,2,2)
%     boxplot([abs(orig_func_plot-pred_obs_plot)'])
%     set(gca,'XTick',[1 2], 'XTickLabel', 'observer')
%     ylim([0 4])
%     set(gca,'FontSize',20)
    
    set(figure(1),'Position',[100 100 1000 600]);
    currFrame = getframe(gcf);
    writeVideo(vidObj_auto_vs_obs,currFrame);
  end
  % end video
end

if strcmp(output,'function')
  close(vidObj_auto_vs_obs);
end  

predict_time = toc;
disp(['Time taken to predict on ' num2str(nsamp_te) ' samples: '...
       num2str(predict_time) ' seconds.'])

     
%% Results Section
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
imagesc(K); colorbar;
t_rks = ['Measurement operator (' meas_type ') for Kernel Observer (RKS) with'...
         num2str(nmeas) ' measurements'];
title(t_rks)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(4);
imagesc(abs(A)); colorbar;
t_rks = ['Dynamics operator (abs) (' meas_type ') for Kernel Observer (RKS) with'...
         num2str(nmeas) ' measurements'];
title(t_rks)
xlabel('Bases')
ylabel('Bases')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

if strcmp(output,'error')
  figure(5);
  plot(te_times, rms_error_te, 'b', 'LineWidth', line_width)
  xlabel('Time step')
  ylabel('Error')
  title('Predicted and actual function error over time (unnormalized)')
  set(gca,'FontSize', font_size)
  set(findall(gcf,'type','text'), 'FontSize', font_size)
end  

set(figure(1), 'Position', [100 100 800 600]);
set(figure(2), 'Position', [100 100 800 600]);
set(figure(3), 'Position', [100 100 800 600]);
set(figure(4), 'Position', [100 100 800 600]);
if strcmp(output,'error')
  set(figure(5), 'Position', [100 100 800 600]);
end  

if save_results == 1
  save_file1 = ['./results/ko_test_dataset_' scheme '_nsamptr_'...
                num2str(nsamp_tr) '_sorted_' num2str(sorted) '_fig1_bandwidth' ];
  save_file2 = ['./results/ko_test_dataset_' scheme '_nsamptr_'...
                num2str(nsamp_tr) '_sorted_' num2str(sorted) '_fig2_noise' ];
  save_file3 = ['./results/ko_test_dataset_' scheme '_nsamptr_'...
                num2str(nsamp_tr) '_sorted_' num2str(sorted) '_fig3_meas_op' ];
  save_file4 = ['./results/ko_test_dataset_' scheme '_nsamptr_'...
                num2str(nsamp_tr) '_sorted_' num2str(sorted) '_fig4_dyn_op' ];
  
  set(figure(1),'PaperOrientation','portrait','PaperSize', [8.5 7],...
      'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(2),'PaperOrientation','portrait','PaperSize', [8.5 7],...
      'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(3),'PaperOrientation','portrait','PaperSize', [8.5 7],...
      'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(4),'PaperOrientation','portrait','PaperSize', [8.5 7],...
      'PaperPositionMode', 'auto', 'PaperType','<custom>');
  
  saveas(figure(1), save_file1, ext)
  saveas(figure(2), save_file2, ext)
  saveas(figure(3), save_file3, ext)
  saveas(figure(4), save_file4, ext)
  
  if strcmp(output,'error')
    save_file5 = ['./results/ko_test_dataset_' scheme '_nsamptr_'...
                num2str(nsamp_tr) '_sorted_' num2str(sorted) '_fig5_error' ];
    set(figure(5),'PaperOrientation','portrait','PaperSize', [8.5 7], ...
      'PaperPositionMode', 'auto', 'PaperType','<custom>');
    saveas(figure(5), save_file5, ext)
  end  
end
