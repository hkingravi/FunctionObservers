%========================= rbfnetwork_example =============================
%  
%  This code demonstrates the RBFNetwork class on a basic example. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%
%========================= rbfnetwork_example =============================
%
%  Name:	rbfnetwork_example.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2015/10/16
%  Modified: 2016/03/14
%
%========================= rbfnetwork_example =============================
clc; clear; close all

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('./data')
addpath('./utils')

% plot parameters 
f_lwidth = 3; 
f_marksize = 5;
c_marksize = 15;
font_size = 15;
save_results = 1; 
ext = 'png';

% flags for plotting and saving
save_unit_test = 0; 
plot_nll = 0;

% load previously existing data 
load KRR_test 
data = x;
obs = y_n;

%% Inference
% generate RKS map
nbases = 200;
ndim = 1; 
k_type = 'gaussian';
bandwidth = 1; 
noise = 0.01;
optimizer = struct('method', 'likelihood', 'solver', 'primal', ...
                   'Display', 'on', 'DerivativeCheck', 'off',...
                   'sort_mat', 1);
rks = kernelObserver.RandomKitchenSinks(nbases, ndim, k_type, ...
                                        bandwidth, noise, ...
                                        optimizer);

tic
rks.fit(data, obs); % train in batch 
estimate_time = toc;
pred_data = rks.predict(data);
Kp = rks.transform(data);
Kp_deriv = rks.get_deriv(data);
disp(['Training time for batch: ' num2str(estimate_time)])

K = Kp'*Kp;
K_deriv = Kp_deriv{1}'*Kp_deriv{1};

%% Results
figure(1);
imagesc(K)
title('Kernel Matrix')

figure(2);
imagesc(K_deriv)
title('Kernel Derivative Matrix')

figure(3);
plot(data, obs, 'ro', 'MarkerSize', f_marksize, 'LineWidth', f_lwidth);
hold on; 
plot(data, pred_data, 'g', 'LineWidth', f_lwidth);
h_legend = legend('observations','estimate');
set(h_legend, 'FontSize', font_size);  

if plot_nll ~= 0
  % evaluate nll on multiple inputs
  nbands = 50;
  nnoise = 50;
  bandwidth = linspace(0.01, 2, nbands);
  noise = linspace(0.01, 2, nnoise);
  lik_vals = zeros(nbands, nnoise);
  seed = 0;
  solver_type = 'dual';
  
  tic
  for i=1:nbands
    for j=1:nnoise
      param_vec = log([bandwidth(i); noise(i)]);
      lik_vals(i, j) = kernelObserver.negative_log_likelihood_rks(param_vec, nbases, ndim, seed,...
        data, obs, k_type,...
        'RandomKitchenSinks', solver_type);
    end
  end
  nll_estimate_time = toc;
  disp(['Training time for generating picture of negative log-likelihood: '...
    num2str(nll_estimate_time)])
  
  figure(4);
  surf(bandwidth, noise, lik_vals)
  xlabel('Bandwidth')
  ylabel('Noise')
  zlabel('Negative log-likelihood')
  title('Negative log-likelihood Surface')
  legend('NLL')
  
end

set(figure(1), 'Position', [100 100 800 600]);
set(figure(2), 'Position', [100 100 800 600]);
set(figure(3), 'Position', [100 100 800 600]);

if save_results == 1
  save_file1 = './results/rks_gp_example_fig1_kmat';
  save_file2 = './results/rks_gp_example_fig2_kmat_deriv';
  save_file3 = './results/rks_gp_example_fig3_infer';
  
  set(figure(1),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(2),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(3),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');  
  saveas(figure(1), save_file1, ext)
  saveas(figure(2), save_file2, ext)
  saveas(figure(3), save_file3, ext)
end


% save data for unit testing if necessary
if save_unit_test ~= 0
  disp('Saving data for unit test...')
  save_file = '../tests/data/RKSTest.mat';
  pred_data_expected = pred_data;
  K_expected = K; 
  save(save_file, 'pred_data_expected', 'K_expected')  
end  