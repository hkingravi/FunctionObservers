%====================== rbf_network_log_lik_check =========================
%  
%  This code demonstrates the log_lik class on a basic example. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%    - Hassan A. Kingravi - personal derivations
%
%====================== rbf_network_log_lik_check =========================
%
%  Name:	rbf_network_log_lik_check.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/04/10
%  Modified: 2016/04/10
%
%====================== rbf_network_log_lik_check =========================
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

%% dual solutions
% generate network: try different initializations of network: Init 1
k_type = 'gaussian';
mapper_type = 'RBFNetwork';
bandwidth = 0.1; 
noise = 0.01;
basis = -5:0.6:5;
solver_dual = 'dual';
ncent = size(basis, 2);
params_ds = log([bandwidth; noise]);

disp('Initial parameters: ')
disp(num2str(exp(params_ds)))

% minFunc parameters
options.Display = 'on';
options.useMex = 0;
tic;
opt_params_ds = minFunc(@kernelObserver.negative_log_likelihood, ...
                        params_ds, options, basis, data, obs, k_type, ...
                        mapper_type, solver_dual);                   
t_init_ds = toc;
opt_params_ds = exp(opt_params_ds);
disp('Final parameters using dual: ')
disp(num2str(opt_params_ds))
disp(['Inference time: ' num2str(t_init_ds)])

% construct RBFNetwork solution
mapper_ds = kernelObserver.FeatureMap(mapper_type);
k_obj_ds = kernelObserver.kernelObj(k_type, opt_params_ds(1));
map_struct_ds.kernel_obj = k_obj_ds;
map_struct_ds.centers = basis; 
mapper_ds.fit(map_struct_ds);
m_data_ds = mapper_ds.transform(data);

K_ds = m_data_ds*transpose(m_data_ds) + exp(opt_params_ds(2))*eye(ncent);
weights_ds = K_ds\(m_data_ds*transpose(obs));
pred_data_ds = transpose(weights_ds)*m_data_ds;

% Init 2
bandwidth_large = 1.2; 
noise_large = 1;
params_dl = log([bandwidth_large; noise_large]);
disp('Initial parameters: ')
disp(num2str(exp(params_ds)))

% minFunc parameters
options.Display = 'on';
options.useMex = 0;
tic;
opt_params_dl = minFunc(@kernelObserver.negative_log_likelihood, ...
                        params_dl, options, basis, data, obs, k_type, ...
                        mapper_type, solver_dual);                   
t_init_dl = toc;
opt_params_dl = exp(opt_params_dl);
disp('Final parameters using dual: ')
disp(num2str(opt_params_dl))
disp(['Inference time: ' num2str(t_init_dl)])

% construct RBFNetwork solution
mapper_dl = kernelObserver.FeatureMap(mapper_type);
k_obj_dl = kernelObserver.kernelObj(k_type, opt_params_dl(1));
map_struct_dl.kernel_obj = k_obj_dl;
map_struct_dl.centers = basis; 
mapper_dl.fit(map_struct_dl);
m_data_dl = mapper_dl.transform(data);

K_dl = m_data_dl*transpose(m_data_dl) + exp(opt_params_dl(2))*eye(ncent);
weights_dl = K_dl\(m_data_dl*transpose(obs));
pred_data_dl = transpose(weights_dl)*m_data_dl;

%% primal solutions
% Init 1
solver_primal = 'primal';
params_ps = log([bandwidth; noise]);

disp('Initial parameters: ')
disp(num2str(exp(params_ps)))

% minFunc parameters
tic;
opt_params_ps = minFunc(@kernelObserver.negative_log_likelihood, ...
                        params_ps, options, basis, data, obs, k_type, ...
                        mapper_type, solver_primal);                   
t_init_ps = toc;
opt_params_ps = exp(opt_params_ps);
disp('Final parameters using primal: ')
disp(num2str(opt_params_ps))
disp(['Inference time: ' num2str(t_init_ps)])

% construct RBFNetwork solution
mapper_ps = kernelObserver.FeatureMap(mapper_type);
k_obj_ps = kernelObserver.kernelObj(k_type, opt_params_ps(1));
map_struct_ps.kernel_obj = k_obj_ps;
map_struct_ps.centers = basis; 
mapper_ps.fit(map_struct_ps);
m_data_ps = mapper_ps.transform(data);

K_ps = m_data_ps*transpose(m_data_ps) + exp(opt_params_ps(2))*eye(ncent);
weights_ps = K_ds\(m_data_ps*transpose(obs));
pred_data_ps = transpose(weights_ps)*m_data_ps;

% Init 2
bandwidth_large = 1.2; 
noise_large = 1;
params_pl = log([bandwidth_large; noise_large]);
disp('Initial parameters: ')
disp(num2str(exp(params_ds)))

% minFunc parameters
options.Display = 'on';
options.useMex = 0;
tic;
opt_params_pl = minFunc(@kernelObserver.negative_log_likelihood, ...
                        params_pl, options, basis, data, obs, k_type, ...
                        mapper_type, solver_primal);                   
t_init_pl = toc;
opt_params_pl = exp(opt_params_pl);
disp('Final parameters using dual: ')
disp(num2str(opt_params_pl))
disp(['Inference time: ' num2str(t_init_pl)])

% construct RBFNetwork solution
mapper_pl = kernelObserver.FeatureMap(mapper_type);
k_obj_pl = kernelObserver.kernelObj(k_type, opt_params_pl(1));
map_struct_pl.kernel_obj = k_obj_pl;
map_struct_pl.centers = basis; 
mapper_pl.fit(map_struct_pl);
m_data_pl = mapper_pl.transform(data);

K_pl = m_data_pl*transpose(m_data_pl) + exp(opt_params_pl(2))*eye(ncent);
weights_pl = K_pl\(m_data_pl*transpose(obs));
pred_data_pl = transpose(weights_pl)*m_data_pl;


figure(1);
subplot(1, 2, 1)
imagesc(K_ds)
title('Kernel matrix (dual, small init)')
subplot(1, 2, 2)
imagesc(K_dl)
title('Kernel matrix (dual, large init)')

figure(2);
plot(data, obs, 'ro', 'MarkerSize', f_marksize);
hold on; 
plot(data, pred_data_ds, 'g', 'LineWidth', f_lwidth);
hold on; 
plot(data, pred_data_dl, 'b--', 'LineWidth', f_lwidth);
hold on; 
plot(basis, zeros(1, ncent), 'cd', 'MarkerSize',c_marksize)
h_legend = legend('observations','estimate (small)', 'estimate (large)', ...
                  'centers');
set(h_legend,'FontSize',20);  
title('RBFNetwork parameter solutions using Gaussian Process (dual)')

figure(3);
subplot(1, 2, 1)
imagesc(K_ps)
title('Kernel matrix (dual, small init)')
subplot(1, 2, 2)
imagesc(K_pl)
title('Kernel matrix (dual, large init)')

figure(4);
plot(data, obs, 'ro', 'MarkerSize', f_marksize);
hold on; 
plot(data, pred_data_ps, 'g', 'LineWidth', f_lwidth);
hold on; 
plot(data, pred_data_pl, 'b--', 'LineWidth', f_lwidth);
hold on;
plot(basis, zeros(1, ncent), 'cd', 'MarkerSize',c_marksize)
h_legend = legend('observations','estimate (small)', 'estimate (large)', ...
                  'centers');
set(h_legend,'FontSize',20);  
title('RBFNetwork parameter solutions using Gaussian Process (primal)')
