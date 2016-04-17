%======================= rbf_network_kernel_check =========================
%  
%  This code checks the differences between the feature map generated by 
%  the RBFNetwork class and the feature map generated by the actual kernel
%  space, as well as the derivative of the feature map. 
%
%  References(s):
%    - Chris Bishop -Pattern Recognition and Machine Learning
%                    1st Edition, Chapter 3.3, pgs 152-161. 
%    - Hassan A. Kingravi - personal derivations
%
%======================= rbf_network_kernel_check =========================
%
%  Name:	rbf_network_kernel_check.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/04/10
%  Modified: 2016/04/10
%
%======================= rbf_network_kernel_check =========================
clc; clear; close all

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('../examples/data')
addpath('../examples/utils')

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
nbases = 200;
centers = linspace(-5, 5, nbases);
ncent = length(centers);
k_type = 'gaussian';
bandwidth = 1; 
batch_noise = 0.01;
k_obj = kernelObserver.kernelObj(k_type, bandwidth);
map_struct.centers = centers;
map_struct.kernel_obj = k_obj; 
fmap = kernelObserver.FeatureMap('RBFNetwork');
fmap.fit(map_struct)
m_data = fmap.transform(data);
md_data = fmap.get_deriv(data);

% first, check difference in kernel matrix structure
Kp = m_data'*m_data;
[Kd, Kd_deriv] = kernelObserver.generic_kernel(data, data, k_obj);
Kp_normalized = Kp/max(max(Kp));

figure(1);
imagesc(Kp_normalized); colorbar
title('Normalized primal RBF network kernel matrix')
figure(2);
imagesc(Kd); colorbar
title('RBF kernel matrix')

disp(['Relative difference in norm between matrices: ' num2str(norm(Kp_normalized-Kd))]);

% check differences in derivatives 
Kp_deriv = md_data{1}'*m_data + m_data'*md_data{1};
figure(3);
imagesc(Kp_deriv); colorbar
title('Normalized primal RBF network kernel derivative matrix')
figure(4);
imagesc(Kd_deriv{1}); colorbar
title('RBF kernel derivative matrix')


