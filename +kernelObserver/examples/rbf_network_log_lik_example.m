%===================== rbfnetwork_example_log_lik =========================
%  
%  This code generates the picture of the negative log-likelihood function 
%  for the RBFNetwork class on a basic example, for various parameter
%  settings. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%
%===================== rbfnetwork_example_log_lik =========================
%
%  Name:	rbfnetwork_example.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2015/10/16
%  Modified: 2016/03/14
%
%===================== rbfnetwork_example_log_lik =========================
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
                   'Display', 'on');
rbfn = kernelObserver.RBFNetwork(centers, k_type, bandwidth, ...
                                 batch_noise, optimizer);

nbands = 100;
nnoise = 100;
bandwidth = linspace(0.1, 2, nbands);
noise = linspace(1, 2, nnoise);

lik_vals = zeros(nbands, nnoise);

tic

for i=1:nbands
  for j=1:nnoise
    param_vec = log([bandwidth(i); noise(i)]);
    lik_vals(i, j) = rbfn.negloglik(param_vec, data, obs);
  end
end  

batch_estimate_time = toc;
disp(['Training time for generating picture of negative log-likelihood: '...
      num2str(batch_estimate_time)])

figure(1);
surf(bandwidth, noise, lik_vals)
xlabel('Bandwidth')
ylabel('Noise')
zlabel('Negative log-likelihood')
title('Negative log-likelihood Surface')
