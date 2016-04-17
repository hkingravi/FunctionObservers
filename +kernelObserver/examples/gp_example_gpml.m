%========================== gp_example_gpml ===============================
%  
%  Given regression problem and kernel, perform hyperparameter optimization
%  and function estimation using Gaussian process regression, using GPML
%  toolbox. 
%
%  References(s):
%    - Rasmussed and Williams - GPs for ML. 
%
%========================== gp_example_gpml ===============================
%
%  Name:	gp_example_gpml.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/04/05
%  Modified: 2016/04/05
%
%========================== gp_example_gpml ===============================
clc; clear; close all

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('./data')
addpath('./utils')

% add GPML toolbox available
run('../../../../gpml-matlab-v3.6-2015-07-07/startup.m')

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
batch_noise = 0.01;

% set up GP
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, data', obs');
nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, data', obs');
[m, s2] = gp(hyp2, @infExact, [], covfunc, likfunc, data', obs', data');

pred_data = m';

disp('Final chosen parameters: covariance params: ')
disp(num2str(hyp2.cov))
disp(['Noise: ' num2str(exp(hyp2.lik))])

figure(1);
plot(data, obs, 'ro', 'MarkerSize', f_marksize);
hold on; 
plot(data, pred_data , 'g', 'LineWidth', f_lwidth);
hold on; 
plot(centers, zeros(1,ncent), 'cd', 'MarkerSize', c_marksize)
h_legend = legend('observations','estimate','centers');
set(h_legend,'FontSize',20);  

% evaluate nll on multiple inputs
nbands = 100;
nnoise = 100;
bandwidth = linspace(0.01, 2, nbands);
noise = linspace(0.01, 2, nnoise);
lik_vals = zeros(nbands, nnoise);

tic
for i=1:nbands
  for j=1:nnoise
    hyp2.cov(1) = bandwidth(i);
    hyp2.lik = noise(j);
    lik_vals(i, j) = gp(hyp2, @infExact, [], covfunc, likfunc, data', obs');
  end
end  
batch_estimate_time = toc;
disp(['Training time for generating picture of negative log-likelihood: '...
      num2str(batch_estimate_time)])

figure(2);
surf(bandwidth, noise, lik_vals')
xlabel('Bandwidth')
ylabel('Noise')
zlabel('Negative log-likelihood')
title('Negative log-likelihood Surface')
legend('NLL')



