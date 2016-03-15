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

% generate network 
centers = -5:0.6:5;
ncent = length(centers);
k_type = 'gaussian';
bandwidth = [0.7, 0.8, 1, 1.2, 1.5]; 
batch_noise = [0.1, 0.2, 0.5, 1];
optimizer = struct('method', 'cv', 'nfolds', 5);
rbfn = kernelObserver.RBFNetwork(centers, k_type, bandwidth, ...
                                 batch_noise, optimizer);

tic
rbfn.fit(data, obs); % train in batch 
batch_estimate_time = toc;
disp(['Training time for batch: ' num2str(batch_estimate_time)])

pred_data = rbfn.predict(data);
K = rbfn.transform(data);

% plot to see estimates 
figure(1);
imagesc(K)
ylabel('Centers')
xlabel('Data')
title('Kernel Matrix')

figure(2);
plot(data,obs,'ro','MarkerSize',f_marksize);
hold on; 
plot(data,pred_data,'g','LineWidth',f_lwidth);
hold on; 
plot(centers,zeros(1,ncent),'cd','MarkerSize',c_marksize)
h_legend = legend('observations','estimate','centers');
set(h_legend,'FontSize',20);  

% save data for unit testing if necessary
if save_unit_test ~= 0
  disp('Saving data for unit test...')
  save_file = '../tests/data/RBFNetworkCVTest.mat';
  pred_data_expected = pred_data;
  K_expected = K; 
  save(save_file, 'pred_data_expected', 'K_expected')  
end  
