%========================= rbfnetwork_example =============================
%  
%  This code demonstrates the RBFNetwork class on a basic example. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%
%========================= rbfnetwork_example =============================
%
%  Name:	rbfnetwork_example.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2015/10/16
%  Modified: 2016/03/14
%
%========================= rbfnetwork_example =============================
clc; clear; close all

%% Step 1: initial setup
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
c_marksize = 15;
font_size = 15;
save_results = 1; 
ext = 'png';

% save data for unit test
save_unit_test = 0; 

% set seed
seed = 2; 
s = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(s);

%% Step 2: construct function observations
[X_cent, Y_cent] = meshgrid(-2:0.5:2, -2:0.5:2);
ncent = size(X_cent, 1)*size(X_cent, 2);
centers = [reshape(X_cent, 1, ncent); reshape(Y_cent, 1, ncent)];
weights = randn(ncent, 1);
[X, Y] = meshgrid(-3:0.1:3, -3:0.1:3);
nsamp = size(X, 1)*size(X, 2);
data = [reshape(X, 1, nsamp); reshape(Y, 1, nsamp)];
params_actual = [1, 0.5, 1.5];  % parameter vector: 2 values per RBF dimension, 
                                % and one scaling argument

Kmat = kernelObserver.sqexp_kernel(centers, data, params_actual);
f_vals = weights'*Kmat; 
Z = reshape(f_vals, size(X, 1), size(X, 2));
obs = f_vals + randn(1, nsamp);  % create observations corrupted by noise
Z_noise = reshape(obs, size(X, 1), size(X, 2));

figure(1);
imagesc(Z); colorbar;
title_str = ['Function generated using squared-exponential kernel with $\ell_1$ = '...
             num2str(params_actual(1)) ', $\ell_2$ = '...
             num2str(params_actual(2))];
title(title_str, 'Interpreter','Latex', 'FontSize', font_size)
set(figure(1), 'Position', [100 100 800 600]);

figure(2);
imagesc(Z_noise); colorbar;
title_str = 'Generated function corrupted by noise for input into RBFNEtwork';
title(title_str, 'Interpreter','Latex', 'FontSize', font_size)



%% generate network using incorrect kernel parameters
k_type = 'sqexp';
params_incorrect = [0.5, 2, 0.6];  % parameter vector: 2 values per RBF dimension, 
                                   % and one scaling argument
batch_noise = 0.01;
optimizer = struct('method', 'likelihood', 'solver', 'primal', ...
                   'Display', 'off', 'DerivativeCheck', 'off');
rbfn_large = kernelObserver.RBFNetwork(centers, k_type, params_incorrect, ...
                                       batch_noise, optimizer);

tic
rbfn_large.fit(data, obs); % train in batch 
est_time = toc;
disp(['Training time for sqexp kernel RBFNetwork with ' num2str(ncent) ...
      ' centers and ' num2str(nsamp) ' samples: '])
disp([num2str(est_time) ' seconds.'])    

tic
pred_data_large = rbfn_large.predict(data);
K_large = rbfn_large.transform(data);
pred_time = toc; 
disp(['Prediction and transform time for sqexp kernel RBFNetwork with ' num2str(ncent) ...
      ' centers and ' num2str(nsamp) ' samples: '])
disp([num2str(pred_time) ' seconds.'])    

% plot to see estimates 
Z_est = reshape(pred_data_large, size(X, 1), size(X, 2));
figure(3);
imagesc(Z_est); colorbar;
title_str = 'Estimated function';
title(title_str, 'Interpreter','Latex', 'FontSize', font_size)


%% display/store results
set(figure(1), 'Position', [100 100 800 600]);
set(figure(2), 'Position', [100 100 800 600]);
set(figure(3), 'Position', [100 100 800 600]);

if save_results == 1
  save_file1 = './results/rbfn_two_dim_example_orig_func';
  save_file2 = './results/rbfn_two_dim_example_orig_noisy';
  save_file3 = './results/rbfn_two_dim_example_est_func';
  
  set(figure(1),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(2),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(3),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  saveas(figure(1), save_file1, ext)
  saveas(figure(2), save_file2, ext)
  saveas(figure(3), save_file3, ext)
end
% 
% % save data for unit testing if necessary
% if save_unit_test ~= 0
%   disp('Saving data for unit test...')
%   save_file = '../tests/data/RBFNetworkTest.mat';
%   pred_data_expected = pred_data;
%   K_expected = K; 
%   save(save_file, 'pred_data_expected', 'K_expected')  
% end  
