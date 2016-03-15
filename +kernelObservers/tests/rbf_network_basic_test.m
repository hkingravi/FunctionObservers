%======================== rbfnetwork_basic_test ===========================
%  
%  This code runs a unit test to check that the RBFNetwork class behaves as
%  expected. If there is nothing displayed after running this code, 
%
%======================== rbfnetwork_basic_test ===========================
%
%  Name:	rbfnetwork_basic_test.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2015/03/14
%  Modified: 2015/03/14
%
%======================== rbfnetwork_basic_test ===========================
clc; clear; close all

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('../examples/data')
addpath('../examples/utils')
addpath('./data')

% load previously existing data 
load KRR_test 
data = x;
obs = y_n;

% load data to check against
load RBFNetworkTest
tol = 1e-12;

% generate network 
centers = -5:0.6:5;
ncent = length(centers);
k_type = 'gaussian';
bandwidth = 0.8; 
batch_noise = 0.1;
rbfn = kernelObserver.RBFNetwork(centers, k_type, bandwidth, batch_noise);

tic
rbfn.fit(data,obs); % train in batch 
batch_estimate_time = toc;

%% Test 1: Ensure RBFNetwork regression prediction produces correct result
pred_data = rbfn.predict(data);
try
  assert(norm(pred_data-pred_data_expected) <= tol, 'RBFNetwork: Problem with regression prediction!')
catch ME
  disp([ME.message])
end

%% Test 2: Ensure RBFNetwork feature mapping behaves as expected
K = rbfn.transform(data);
try
  assert(norm(K-K_expected) <= tol, 'RBFNetwork: Problem with feature space mapping!')
catch ME
  disp([ME.message])
end


