%=========================== log_lik_example ==============================
%  
%  This code demonstrates the log_lik class on a basic example. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%
%=========================== log_lik_example ==============================
%
%  Name:	log_lik_example.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2015/03/27
%  Modified: 2016/03/27
%
%=========================== log_lik_example ==============================
clc; clear; close all

% add path to kernelObserver folder and data
if ispc == 1kernelObserver.solve_chol(L, eye(nsamp))/sl
  addpath('../')
else
  addpath('../../')
end  
addpath('./data')
addpath('./utils')
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

%% dual solutions
% generate network: try different initializations of network: Init 1
k_type = 'gaussian';
mapper_type = 'RBFNetwork';
bandwidth = 0.1; 
noise = 0.01;
basis = -5:0.6:5;
solver_dual = 'dual';
ncent = size(basis, 2);
params = log([bandwidth; noise]);

disp('Initial parameters: ')
disp(num2str(exp(params)))

% minFunc parameters
options.Display = 'on';
options.useMex = 0;
tic;
opt_params = minFunc(@kernelObserver.negative_log_likelihood, ...
                     params, options, basis, data, obs, k_type, ...
                     mapper_type, solver_dual);                   
t_init_1_dual = toc;
disp('Final parameters using dual: ')
disp(num2str(exp(opt_params)))
disp(['Inference time: ' num2str(t_init_1_dual)])


% construct RBFNetwork solution
mapper = kernelObserver.FeatureMap(mapper_type);
k_obj = kernelObserver.kernelObj(k_type, opt_params(1));
map_struct.kernel_obj = k_obj;
map_struct.centers = basis; 
mapper.fit(map_struct);
m_data = mapper.transform(data);

K = m_data*transpose(m_data) + exp(opt_params(2))*eye(ncent);
weights = K\(m_data*transpose(obs));
pred_data_small = transpose(weights)*m_data;

% Init 2
bandwidth_large = 1.2; 
noise_large = 1;
params = log([bandwidth_large; noise_large]);
disp('Initial parameters: ')
disp(num2str(exp(params)))

% minFunc parameters
options.Display = 'on';
options.useMex = 0;
tic;
opt_params = minFunc(@kernelObserver.negative_log_likelihood, ...
                     params, options, basis, data, obs, k_type, ...
                     mapper_type, solver_dual);                   
t_init_1_dual = toc;
disp('Final parameters using dual: ')
disp(num2str(exp(opt_params)))
disp(['Inference time: ' num2str(t_init_1_dual)])

% construct RBFNetwork solution
mapper = kernelObserver.FeatureMap(mapper_type);
k_obj = kernelObserver.kernelObj(k_type, opt_params(1));
map_struct.kernel_obj = k_obj;
map_struct.centers = basis; 
mapper.fit(map_struct);
m_data = mapper.transform(data);

K = m_data*transpose(m_data) + exp(opt_params(2))*eye(ncent);
weights = K\(m_data*transpose(obs));
pred_data_large = transpose(weights)*m_data;

%% primal solution
% generate network: try different initializations of network: Init 1
bandwidth = 0.1; 
noise = 0.01;
solver_primal = 'primal';
params = log([bandwidth; noise]);

disp('Initial parameters: ')
disp(num2str(exp(params)))

% minFunc parameters
options.Display = 'on';
options.useMex = 0;
tic
opt_params = minFunc(@kernelObserver.negative_log_likelihood, ...
                     params, options, basis, data, obs, k_type, ...
                     mapper_type, solver_primal);
t_init_1_primal = toc;
disp('Final parameters using primal: ')
disp(num2str(exp(opt_params)))
disp(['Inference time: ' num2str(t_init_1_primal)])

% construct RBFNetwork solution
mapper = kernelObserver.FeatureMap(mapper_type);
k_obj = kernelObserver.kernelObj(k_type, opt_params(1));
map_struct.kernel_obj = k_obj;
map_struct.centers = basis; 
mapper.fit(map_struct);
m_data = mapper.transform(data);

K = m_data*transpose(m_data) + exp(opt_params(2))*eye(ncent);
weights = K\(m_data*transpose(obs));
pred_data_small_p = transpose(weights)*m_data;

% Init 2
params = log([bandwidth_large; noise_large]);
disp('Initial parameters: ')
disp(num2str(exp(params)))

% minFunc parameters
options.Display = 'on';
options.useMex = 0;
tic
opt_params = minFunc(@kernelObserver.negative_log_likelihood, ...
                     params, options, basis, data, obs, k_type, ...
                     mapper_type, solver_primal);                   
t_init_2_primal = toc;
disp('Final parameters using primal: ')
disp(num2str(exp(opt_params)))
disp(['Inference time: ' num2str(t_init_2_primal)])

% construct RBFNetwork solution
mapper = kernelObserver.FeatureMap(mapper_type);
k_obj = kernelObserver.kernelObj(k_type, opt_params(1));
map_struct.kernel_obj = k_obj;
map_struct.centers = basis; 
mapper.fit(map_struct);
m_data = mapper.transform(data);

K = m_data*transpose(m_data) + exp(opt_params(2))*eye(ncent);
weights = K\(m_data*transpose(obs));
pred_data_large_p = transpose(weights)*m_data;

figure(1);
plot(data, obs, 'ro', 'MarkerSize', f_marksize);
hold on; 
plot(data, pred_data_small, 'g', 'LineWidth', f_lwidth);
hold on; 
plot(data, pred_data_large, 'b--', 'LineWidth', f_lwidth);
hold on; 
plot(basis, zeros(1, ncent), 'cd', 'MarkerSize',c_marksize)
h_legend = legend('observations','estimate (small)', 'estimate (large)', ...
                  'centers');
set(h_legend,'FontSize',20);  
title('RBFNetwork parameter solutions using Gaussian Process (dual)')

figure(2);
plot(data, obs, 'ro', 'MarkerSize', f_marksize);
hold on; 
plot(data, pred_data_small_p, 'g', 'LineWidth', f_lwidth);
hold on; 
plot(data, pred_data_large_p, 'b--', 'LineWidth', f_lwidth);
hold on; 
plot(basis, zeros(1, ncent), 'cd', 'MarkerSize',c_marksize)
h_legend = legend('observations','estimate (small)', 'estimate (large)', ...
                  'centers');
set(h_legend,'FontSize',20);  
title('RBFNetwork parameter solutions using Gaussian Process (primal)')
