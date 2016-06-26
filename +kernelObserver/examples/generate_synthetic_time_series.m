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
seed = 30; 
s = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(s);

% plot parameters
f_lwidth = 3; 
c_marksize = 15;
font_size = 15;
save_results = 1; 
ext = 'png';
use_plot_min = 1;

%% time-varying data arises from weight set associated to an RBF network
eval_data = 0:0.005:2*pi; % where the function values are evaluated for plotting purposes 
orig_function_dim = 5;
centers_orig = linspace(0, 2*pi, orig_function_dim); % we assume that these are shared 
generator = 'RBFNetwork';
scheme = 'smooth3'; % smoothly varying system 
sys_type = 'continuous'; 

% scheme plotting limits
xmin = -0.1;
xmax = 2*pi+0.1;

if strcmp(scheme, 'switching') || strcmp(scheme, 'switching2') || ...
   strcmp(scheme, 'smooth3') 
  ymin = -4;
  ymax = 4;
elseif strcmp(scheme, 'smooth2')
  ymin = -10;
  ymax = 10;
end

% create rbf network to generate the time-varying function
k_type = 'gaussian';
bandwidth = 0.3; % pass in multiple bandwidths to check correct handling
noise = 0.1; % pass in multiple noise parameters to check correct handling
model = kernelObserver.RBFNetwork(centers_orig, k_type, bandwidth, noise); 

init_weights_orig = randn(orig_function_dim, 1);
weights = init_weights_orig;
dt = 0.03; 
current_t = 0; 
final_time = 20;
num_steps = floor(final_time/dt); 

nsamp = 500;
nsamp_plot = length(eval_data); % number of samples for plotting
data = linspace(0, 2*pi, nsamp); % where the function values are sampled from

%% Functional Output Selection 
% select which type of output you're interested in seeing; default is the
% tracking error, while the other option is visualization of the functions
create_dynamic_output = 1; 
frames_to_skip = 2; 
output = 'gif'; % 'gif' or 'function'

if strcmp(output,'function')
  % create gif
  vidObj_auto_vs_obs = VideoWriter(['./results/gen_synth_ts_' scheme '.avi'] );
  open(vidObj_auto_vs_obs);
elseif strcmp(output,'gif')
  gif_file = ['./results/gen_synth_ts_' scheme '.gif'];
end

%% run simulation
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
  seed = i; 
  s = RandStream('mt19937ar','Seed',seed);
  RandStream.setGlobalStream(s);
  
  weights = time_varying_uncertainty(weights, times(i), scheme);  % get weights
  
  model.set('weights', weights);
  orig_func = model.predict(data); 
  orig_func_plot = model.predict(eval_data); 
  orig_obs = orig_func + noise*randn(1, nsamp);
    
  % store function values for error plots later   
  orig_func_vals(:, :, i) = orig_func;  
  orig_func_plot_vals(:, :, i) = orig_func_plot;  
  orig_func_data(:, :, i) = data;  
  orig_func_obs(:, :, i) = orig_obs;  
  ideal_weight_trajectory(:, i) = weights;
end

% generate image 
title_str = ['Synthetic time series using ' scheme ' scheme (weights)'];
figure(1);
plot(times, ideal_weight_trajectory', 'LineWidth', 1.5)
ylabel('Weights', 'FontSize', font_size)
xlabel('Time', 'FontSize', font_size)
xlim([0, max(times)])
title(title_str, 'FontSize', font_size)
set(gca, 'FontSize', font_size)

ymin_weights = min(min(ideal_weight_trajectory));
ymax_weights = max(max(ideal_weight_trajectory));

% generate image 
title_str = ['Synthetic time series using ' scheme ' scheme (function)'];
figure(2);
imagesc(squeeze(orig_func_plot_vals(1, :, :)))
ylabel('Domain', 'FontSize', font_size)
xlabel('Time', 'FontSize', font_size)
set(gca, 'XTickLabel', '', 'YTickLabel', '')
title(title_str, 'FontSize', font_size)
set(gca, 'FontSize', font_size)

% vid output
vid_start = 1;
vid_end = num_steps; 

if create_dynamic_output == 1
  nroll = 100; y_vals = linspace(-10, 10, nroll);  % create rolling bar
  first_frame_flag = 0; 
  for i=vid_start:vid_end
    if mod(i, frames_to_skip) == 0
      figure(3);
      % first plot is the functional output
      subplot(1, 2, 1)
      plot(eval_data, squeeze(orig_func_plot_vals(:, :, i)), 'g-',...
        'LineWidth', f_lwidth);
      hold on;
      curr_obs = squeeze(orig_func_obs(:, :, i)) + ...
                 noise*randn(1, size(orig_func_data(:, :, i), 2));
      plot(squeeze(orig_func_data(:, :, i)), curr_obs, 'ro', 'LineWidth', 1.3);
      hold on;
      plot(centers_orig, zeros(1, orig_function_dim), 'bd', ...
        'MarkerSize', c_marksize, 'LineWidth', f_lwidth);
      ylabel('Function', 'FontSize', font_size)
      xlabel('Domain', 'FontSize', font_size)
      h_legend = legend('function', 'observations', 'centers');
      set(h_legend,'FontSize', font_size);
      set(gca, 'FontSize', font_size)
      set(figure(3), 'Position', [100 100 1400 600]);
      hold off
      if use_plot_min == 1
        xlim([xmin xmax])
        ylim([ymin ymax])
      end
      
      % second plot is the current values of the weights
      subplot(1, 2, 2)
      plot(times, ideal_weight_trajectory', 'LineWidth', 1.5)
      ylabel('Weights', 'FontSize', font_size)
      xlabel('Time (seconds)', 'FontSize', font_size)
      hold on;
      x_vals = times(i)*ones(1, nroll);
      plot(x_vals, y_vals, 'g-', 'LineWidth', f_lwidth)
      hold off;
      set(gca, 'FontSize', font_size)
      if use_plot_min == 1
        xlim([0, times(end)])
        ylim([ymin_weights ymax_weights])
      end
      
      if strcmp(output, 'function')
        currFrame = getframe(gcf);
        writeVideo(vidObj_auto_vs_obs,currFrame);
      else
        drawnow;
        frame = getframe(figure(3));
        [A_g, map] = rgb2ind(frame.cdata, 256, 'nodither');
        if first_frame_flag == 0
          imwrite(A_g, map, gif_file, 'gif', 'LoopCount', Inf, 'DelayTime', 0);
          first_frame_flag = 1;
        else
          imwrite(A_g, map, gif_file, 'gif', 'WriteMode', 'append', 'DelayTime', 0);
        end
      end % end output
    end % end frame skipper
   
  end % end loop
  if strcmp(output,'function')
    close(vidObj_auto_vs_obs);
  end
end

if save_results == 1
  save_file1 = './results/gen_synth_ts_fig1_weights';
  save_file2 = './results/gen_synth_ts_fig2_func';
  
  set(figure(1), 'Position', [100 100 800 600]);
  set(figure(2), 'Position', [100 100 800 600]);
  saveas(figure(1), save_file1, ext)
  saveas(figure(2), save_file2, ext)  
end

% save observations to a file
save_file = ['./data/synthetic_time_series_generator_' generator ...
             '_kernel_' k_type '_scheme_' scheme '.mat'];
save(save_file, 'orig_func_data', 'orig_func_obs',...
     'orig_func_plot_vals', 'times')