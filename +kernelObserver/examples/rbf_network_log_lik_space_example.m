%================== rbf_network_log_lik_space_example =====================
%  
%  This code generates the picture of the negative log-likelihood function 
%  for the RBFNetwork class on a basic example, for various parameter
%  settings. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%
%================== rbf_network_log_lik_space_example =====================
%
%  Name:	rbf_network_log_lik_space_example.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2015/10/16
%  Modified: 2016/03/14
%
%================== rbf_network_log_lik_space_example =====================
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
solver_type = 'primal';

disp(['Solving RBFNetwork in the ' solver_type])
optimizer = struct('method', 'likelihood', 'solver', solver_type,...
                   'Display', 'on');
rbfn = kernelObserver.RBFNetwork(centers, k_type, bandwidth,...
                                 batch_noise, optimizer);
tic
rbfn.fit(data, obs);
opt_time = toc;
params_final = rbfn.get_params();
disp(['Training time for optimizing RBFN: '...
      num2str(opt_time)])    
disp('Optimal parameters:')
disp(num2str(params_final))
                               
nbands = 50;
nnoise = 50;
bandwidth = linspace(0.1, 2, nbands);
noise = linspace(0.05, 2, nnoise);

lik_vals = zeros(nbands, nnoise);

tic

for i=1:nbands
  for j=1:nnoise
    param_vec = log([bandwidth(i); noise(i)]);
    lik_vals(i, j) = kernelObserver.negative_log_likelihood(param_vec, centers,...
                                                            data, obs, k_type,...
                                                            'RBFNetwork', solver_type);
  end
end  

batch_estimate_time = toc;
disp(['Training time for generating picture of negative log-likelihood: '...
      num2str(batch_estimate_time)])

% find closes indices 
[~, id_band] = min(abs(bandwidth - params_final(1)));
[~, id_noise] = min(abs(noise - params_final(2)));

    
figure(1);
surf(bandwidth, noise, lik_vals)
hold on; 
plot3(bandwidth(id_band), noise(id_noise), lik_vals(id_band, id_noise),...
      'ro', 'markers', 12)
xlabel('Bandwidth')
ylabel('Noise')
zlabel('Negative log-likelihood')
title('Negative log-likelihood Surface')
legend('NLL', 'optimal parameter')
