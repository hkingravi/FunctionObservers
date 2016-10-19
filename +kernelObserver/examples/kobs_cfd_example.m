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
clc; close all;clear all;

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('../examples/data')
addpath('../examples/utils')
addpath('./data')

% set random seed
s = RandStream('mcg16807','Seed',101);
RandStream.setGlobalStream(s)

% load fixed basis vector set 
load('basis300_bw0.4.mat')
centers = gp1_BV;
ncent = size(centers,2);

load('allsnaps.mat');
X = X{1}; V = V{1};
data = X';
obs = V(:, 1)';
nsamp = size(data, 2);
nsamp_red = 2000;  % number of points to retain

subsample = 1;

if subsample == 1
  rand_inds = randperm(nsamp);
  data = data(:, rand_inds(1:nsamp_red));
  obs = obs(rand_inds(1:nsamp_red));
end  

% Set up RBFNetwork
k_type = 'sqexp';
params_guess = [0.4, 0.4, 0.9];  % parameter vector: 2 values per RBF dimension, and one scaling argument
batch_noise = 0.01;
optimizer = struct('method', 'likelihood', 'solver', 'primal', ...
                   'Display', 'off', 'DerivativeCheck', 'off');
rbfn_large = kernelObserver.RBFNetwork(centers, k_type, params_guess, batch_noise, optimizer);

tic
rbfn_large.fit(data, obs); % train in batch 
est_time = toc;

% Error using chol
% Matrix must be positive definite.
% 
% Error in kernelObserver.negative_log_likelihood (line 85)
%     L = chol(Kp/noise + eye(ncent)); sl = noise;
% 
% Error in WolfeLineSearch (line 34)
%     [f_new,g_new] = funObj(x+t*d,varargin{:});
% 
% Error in minFunc (line 1067)
%             [t,f,g,LSfunEvals] =
%             WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS_interp,LS_multi,25,progTol,debug,doPlot,1,funObj,varargin{:});
%             
% Error in kernelObserver.RBFNetwork/fit (line 247)
%         opt_params = minFunc(@kernelObserver.negative_log_likelihood, ...
% 
% Error in learn_hyperparams (line 31)
% rbfn_large.fit(data, obs); % train in batch
%  