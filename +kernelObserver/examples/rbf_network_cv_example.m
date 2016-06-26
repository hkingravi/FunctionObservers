%======================== rbfnetwork_cv_example ===========================
%  
%  This code demonstrates the RBFNetwork class on a basic example. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%
%======================== rbfnetwork_cv_example ===========================
%
%  Name:	rbfnetwork_cv_example.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2015/10/16
%  Modified: 2016/03/14
%
%======================== rbfnetwork_cv_example ===========================
clc; clear; close all

%% Setup

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

% save data for unit test
save_unit_test = 0; 

%% Inference
% load previously existing data 
load KRR_test 
data = x;
obs = y_n;

% generate network 
centers = -5:0.6:5;
ncent = length(centers);
k_type = 'gaussian';
bandwidth = linspace(0.05, 3, 100);
batch_noise = linspace(0.05, 2, 100);
optimizer = struct('method', 'cv', 'nfolds', 5);
rbfn = kernelObserver.RBFNetwork(centers, k_type, bandwidth, ...
                                 batch_noise, optimizer);

tic
rbfn.fit(data, obs); % train in batch 
batch_estimate_time = toc;
disp(['Training time for batch: ' num2str(batch_estimate_time)])

pred_data = rbfn.predict(data);
K = rbfn.transform(data);

%% Results Section
figure(1);
imagesc(K)
ylabel('Centers')
xlabel('Data')
title('Kernel Matrix')

figure(2);
plot(data, obs, 'ro', 'MarkerSize', f_marksize, 'LineWidth', f_lwidth);
hold on; 
plot(data, pred_data, 'g', 'LineWidth', f_lwidth);
hold on; 
plot(centers, zeros(1,ncent), 'bd', 'MarkerSize', c_marksize, 'LineWidth', f_lwidth)
h_legend = legend('observations','estimate','centers');
set(h_legend, 'FontSize', font_size);  

set(figure(1), 'Position', [100 100 800 600]);
set(figure(2), 'Position', [100 100 800 600]);

if save_results == 1
  save_file1 = './results/rbfn_cv_example_fig1_kmat';
  save_file2 = './results/rbfn_cv_example_fig2_infer';
  
  set(figure(1),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(2),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  saveas(figure(1), save_file1, ext)
  saveas(figure(2), save_file2, ext)
end

% save data for unit testing if necessary
if save_unit_test ~= 0
  disp('Saving data for unit test...')
  save_file = '../tests/data/RBFNetworkCVTest.mat';
  pred_data_expected = pred_data;
  K_expected = K; 
  save(save_file, 'pred_data_expected', 'K_expected')  
end  
