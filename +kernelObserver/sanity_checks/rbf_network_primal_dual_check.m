%==================== rbf_network_primal_dual_check =======================
%  
%  This code checks whether gradient parameter updates using the primal and 
%  dual formulations are equivalent to each other for the RBFNetwork case. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%    - Hassan A. Kingravi - personal derivations
%
%==================== rbf_network_primal_dual_check =======================
%
%  Name:	rbf_network_primal_dual_check.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/03/27
%  Modified: 2016/04/10
%
%==================== rbf_network_primal_dual_check =======================
clc; clear; close all

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('../examples/data')
addpath('../examples/utils')
addpath('../../minFunc/minFunc/')

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
nsamp = size(data, 2);

% generate network: try different initializations of network: Init 1
k_type = 'gaussian';
mapper_type = 'RBFNetwork';
bandwidth = 0.1; 
noise = 1e-3;
basis = -5:0.2:5;
solver_dual = 'dual';
ncent = size(basis, 2);
params = log([bandwidth; noise]);

disp('Initial parameters: ')
disp(num2str(exp(params)))

kernelObserver.primal_dual_check(params, basis, data, obs, k_type,...
                                 mapper_type, solver_dual);

                   
                   