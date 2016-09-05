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

% construct data via network
centers = [0; 0];
ncent = size(centers, 2);
weights = 1; %randn(ncent, 1);
[X, Y] = meshgrid(-3:0.05:3, -3:0.05:3);
nsamp = size(X, 1)*size(X, 2);
data = [reshape(X, 1, nsamp); reshape(Y, 1, nsamp)];
params = [1, 0.5, 1];  % parameter vector: 2 values per RBF dimension, 
                       % and one scaling argument

Kmat = kernelObserver.sqexp_kernel(centers, data, params);
f_vals = weights'*Kmat; 
Z = reshape(f_vals, size(X, 1), size(X, 2));

figure(1);
imagesc(Z); colorbar;
title_str = ['Squared-exponential kernel with $\ell_1$ = '...
             num2str(params(1)) ', $\ell_2$ = ' num2str(params(2))];
title(title_str, 'Interpreter','Latex', 'FontSize', font_size)
set(figure(1), 'Position', [100 100 800 600]);

