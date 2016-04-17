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
c_marksize = 10;

% save data for unit test
save_unit_test = 0; 

% load previously existing data 
load KRR_test 
data = x;
obs = y_n;

% generate network: try different initializations of network
centers = -5:0.6:5;
ncent = length(centers);
k_type = 'gaussian';
bandwidth = 0.1; 
batch_noise = 0.01;
optimizer = struct('method', 'likelihood', 'solver', 'dual', ...
                   'Display', 'on', 'DerivativeCheck', 'on');
rbfn_large = kernelObserver.RBFNetwork(centers, k_type, bandwidth, ...
                                       batch_noise, optimizer);

tic
rbfn_large.fit(data, obs); % train in batch 
batch_estimate_time = toc;


pred_data_small = rbfn_large.predict(data);
K_small = rbfn_large.transform(data);
disp(['Training time for batch: ' num2str(batch_estimate_time)])

bandwidth = 2.1; 
batch_noise = 0.1;
rbfn_large = kernelObserver.RBFNetwork(centers, k_type, bandwidth, ...
                                       batch_noise, optimizer);

tic
rbfn_large.fit(data, obs); % train in batch 
batch_estimate_time = toc;


pred_data_large = rbfn_large.predict(data);
K_large = rbfn_large.transform(data);
disp(['Training time for batch: ' num2str(batch_estimate_time)])

% plot to see estimates 
figure(1);
imagesc(K_small)
ylabel('Centers')
xlabel('Data')
title('Kernel Matrix (small values for init)')
figure(2);
imagesc(K_large)
ylabel('Centers')
xlabel('Data')
title('Kernel Matrix (large values for init)')

figure(3);
plot(data, obs, 'ro', 'MarkerSize', f_marksize);
hold on; 
plot(data, pred_data_small, 'g', 'LineWidth', f_lwidth);
hold on; 
plot(data, pred_data_large, 'b', 'LineWidth', f_lwidth);
hold on; 
plot(centers, zeros(1, ncent), 'cd', 'MarkerSize',c_marksize)
h_legend = legend('observations','estimate (small)', 'estimate (large)', ...
                  'centers');
set(h_legend,'FontSize',20);  

% save data for unit testing if necessary
if save_unit_test ~= 0
  disp('Saving data for unit test...')
  save_file = '../tests/data/RBFNetworkTest.mat';
  pred_data_expected = pred_data;
  K_expected = K; 
  save(save_file, 'pred_data_expected', 'K_expected')  
end  
