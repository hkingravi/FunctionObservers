%========================== gp_example_native =============================
%  
%  Given regression problem and kernel, perform hyperparameter optimization
%  and function estimation using Gaussian process regression, using code in
%  this toolbox. 
%
%  References(s):
%    - Rasmussed and Williams - GPs for ML. 
%
%========================== gp_example_native =============================
%
%  Name:	gp_example_native.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/04/05
%  Modified: 2016/04/05
%
%========================== gp_example_native =============================
clc; clear; close all

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('./data')
addpath('./utils')
addpath('../../minFunc/minFunc/')
addpath('../../minFunc/autoDif/')

% plot parameters 
f_lwidth = 3; 
f_marksize = 5;
c_marksize = 10;

% flags for plotting and saving
save_unit_test = 0; 
plot_nll = 1;

% load previously existing data 
load KRR_test 
data = x;
obs = y_n;


k_type = 'gaussian';
bandwidth = 1; 
noise = 0.01;
k_obj = kernelObserver.kernelObj(k_type, bandwidth);
params = [log(k_obj.k_params); log(noise)];
options.useMex = 0;
options.Display = 'on';
options.MaxIter = 350;
options.DerivativeCheck = 'off';
opt_params = minFunc(@kernelObserver.negative_log_likelihood_gp, ...
                     params, options, data, obs, ...
                     k_obj.k_name);
k_obj.k_params = exp(opt_params(1));
noise = exp(opt_params(2));

disp('Final chosen parameters: covariance params: ')
disp(num2str(k_obj.k_params))
disp(['Noise: ' num2str(noise)])

% compute GP solution
nsamp = size(data, 2);
[K, deriv_cell] = kernelObserver.generic_kernel(data, data, k_obj);
L = chol(K + noise*eye(nsamp));
alpha = kernelObserver.solve_chol(L, obs');
pred_data = alpha'*K;

figure(1);
plot(data, obs, 'ro', 'MarkerSize', f_marksize);
hold on; 
plot(data, pred_data , 'g', 'LineWidth', f_lwidth);
h_legend = legend('observations', 'estimate');
set(h_legend,'FontSize',20);  

figure(2);
imagesc(K); colorbar
title('Kernel matrix')

figure(3);
imagesc(deriv_cell{1}); colorbar
title('Derivative of Kernel Matrix')

if plot_nll ~= 0
  % evaluate nll on multiple inputs
  nbands = 50;
  nnoise = 50;
  bandwidth = linspace(0.01, 2, nbands);
  noise = linspace(0.01, 2, nnoise);
  lik_vals = zeros(nbands, nnoise);
  
  tic
  for i=1:nbands
    for j=1:nnoise
      param_vec = log([bandwidth(i); noise(i)]);
      lik_vals(i, j) = kernelObserver.negative_log_likelihood_gp(param_vec, data, obs, k_type);
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
