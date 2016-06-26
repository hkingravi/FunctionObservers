%====================== measurement_operator_test =========================
%  
%  This code tests the measurement_operator function.
%
%====================== measurement_operator_test =========================
%
%  Name:	measurement_operator_test.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/05/01
%  Modified: 2016/05/02
%
%====================== measurement_operator_test =========================
clc; clear all; clear classes; close all

% add path to kernelObserver folder and data
if ispc == 1
  addpath('../')
else
  addpath('../../')
end  
addpath('./data')
addpath('./utils')

% plot parameters
font_size = 16;
save_results = 1;
ext = 'png';

% set seed
seed = 20; 
s = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(s);

%% load data
% save data for unit test
save_unit_test = 0; 


%% measurement map parameters
nmeas_small = 10;
nmeas_large = 50; 
meas_type = 'random';

%% kernel parameters
k_type = 'gaussian';
bandwidth = 0.9;
data = linspace(-6, 6, 500);  % data to randomly subsample from

%% generate truly random RBFNetwork feature map and associated measurement operator
model_type_rbfn = 'RBFNetwork';
map_struct_rbfn.kernel_obj = kernelObserver.kernelObj(k_type, bandwidth);
map_struct_rbfn.centers = -5:0.1:5;
fmap_rbfn = kernelObserver.FeatureMap(model_type_rbfn);
fmap_rbfn.fit(map_struct_rbfn);

Kmat_small_rbfn = kernelObserver.measurement_operator(meas_type, nmeas_small, fmap_rbfn, data);
Kmat_large_rbfn = kernelObserver.measurement_operator(meas_type, nmeas_large, fmap_rbfn, data);

%% generate sorted RBFNetwork feature map and associated measurement operator
model_type_rbfn = 'RBFNetwork';
map_struct_rbfn.kernel_obj = kernelObserver.kernelObj(k_type, bandwidth);
map_struct_rbfn.centers = -5:0.1:5;
map_struct_rbfn.sort_mat = 1;
fmap_rbfn = kernelObserver.FeatureMap(model_type_rbfn);
fmap_rbfn.fit(map_struct_rbfn);

Kmat_small_rbfn_sorted = kernelObserver.measurement_operator(meas_type, nmeas_small, fmap_rbfn, data);
Kmat_large_rbfn_sorted = kernelObserver.measurement_operator(meas_type, nmeas_large, fmap_rbfn, data);

%% generate truly random RandomKitchenSinks feature map and associated measurement operator
model_type_rks = 'RandomKitchenSinks';
map_struct_rks.ndim = 1;
map_struct_rks.nbases = 100;
map_struct_rks.seed = seed;
map_struct_rks.sort_mat = 0;
map_struct_rks.kernel_obj = kernelObserver.kernelObj(k_type, bandwidth);
fmap_rks = kernelObserver.FeatureMap(model_type_rks);
fmap_rks.fit(map_struct_rks);

Kmat_small_rks = kernelObserver.measurement_operator(meas_type, nmeas_small, fmap_rks, data);
Kmat_large_rks = kernelObserver.measurement_operator(meas_type, nmeas_large, fmap_rks, data);

%% generate sorted RandomKitchenSinks feature map and associated measurement operator
bandwidth_sorted = 0.5;
model_type_rks_sorted = 'RandomKitchenSinks';
map_struct_rks_sorted.ndim = 1;
map_struct_rks_sorted.nbases = 400;
map_struct_rks_sorted.seed = seed;
map_struct_rks_sorted.sort_mat = 1;
map_struct_rks_sorted.kernel_obj = kernelObserver.kernelObj(k_type, bandwidth_sorted);
fmap_rks_sorted = kernelObserver.FeatureMap(model_type_rks_sorted);
fmap_rks_sorted.fit(map_struct_rks_sorted);

nmeas_large_sorted = 200;
Kmat_small_rks_sorted = kernelObserver.measurement_operator(meas_type, nmeas_small, fmap_rks_sorted, data);
Kmat_large_rks_sorted = kernelObserver.measurement_operator(meas_type, nmeas_large_sorted, fmap_rks_sorted, data);

%% plot final results
figure(1);
imagesc(Kmat_small_rbfn')
t_small_rbfn = ['Measurement operator (' meas_type ') for '...
               model_type_rbfn ' with ' num2str(nmeas_small) ' measurements'];
title(t_small_rbfn)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(2);
imagesc(Kmat_large_rbfn')
t_large_rbfn = ['Measurement operator (' meas_type ') for '...
               model_type_rbfn ' with ' num2str(nmeas_large) ' measurements'];
title(t_large_rbfn)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(3);
imagesc(Kmat_small_rbfn_sorted')
t_large_rbfn = ['Sorted measurement operator (' meas_type ') for '...
               model_type_rbfn ' with ' num2str(nmeas_large) ' measurements'];
title(t_large_rbfn)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)


figure(4);
imagesc(Kmat_large_rbfn_sorted')
t_large_rbfn = ['Sorted measurement operator (' meas_type ') for '...
               model_type_rbfn ' with ' num2str(nmeas_large) ' measurements'];
title(t_large_rbfn)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)


figure(5);
imagesc(Kmat_small_rks')
t_small_rks = ['Measurement operator (' meas_type ') for '...
               model_type_rks ' with ' num2str(nmeas_small) ' measurements'];
title(t_small_rks)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(6);
imagesc(Kmat_large_rks')
t_large_rks = ['Measurement operator (' meas_type ') for '...
               model_type_rks ' with ' num2str(nmeas_large) ' measurements'];
title(t_large_rks)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(7);
imagesc(Kmat_small_rks_sorted')
t_small_rks_sorted = ['Sorted measurement operator (' meas_type ') for '...
                      model_type_rks_sorted ' with ' num2str(nmeas_small) ' measurements'];
title(t_small_rks_sorted)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(8);
imagesc(Kmat_large_rks_sorted')
t_large_rks_sorted = ['Sorted measurement operator (' meas_type ') for '...
                      model_type_rks_sorted ' with ' num2str(nmeas_large_sorted) ' measurements'];
title(t_large_rks_sorted)
xlabel('Basis')
ylabel('Measurements')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

set(figure(1),'Position',[100 100 800 600]);
set(figure(2),'Position',[100 100 800 600]);
set(figure(3),'Position',[100 100 800 600]);
set(figure(4),'Position',[100 100 800 600]);
set(figure(5),'Position',[100 100 800 600]);
set(figure(6),'Position',[100 100 800 600]);
set(figure(7),'Position',[100 100 800 600]);
set(figure(8),'Position',[100 100 800 600]);

if save_results == 1
  save_file1 = './results/mmap_fig1_kmat_small_rbfn';
  save_file2 = './results/mmap_fig2_kmat_large_rbfn';
  save_file3 = './results/mmap_fig3_kmat_small_rbfn_sorted';
  save_file4 = './results/mmap_fig4_kmat_large_rbfn_sorted';
  save_file5 = './results/mmap_fig1_kmat_small_rks';
  save_file6 = './results/mmap_fig2_kmat_large_rks';
  save_file7 = './results/mmap_fig3_kmat_small_rks_sorted';
  save_file8 = './results/mmap_fig4_kmat_large_rks_sorted';  
  
  set(figure(1),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(2),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(3),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(4),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(5),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(6),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');  
  set(figure(7),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');  
  set(figure(8),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');  
  
  saveas(figure(1), save_file1, ext)
  saveas(figure(2), save_file2, ext)
  saveas(figure(3), save_file3, ext)
  saveas(figure(4), save_file4, ext)
  saveas(figure(5), save_file5, ext)
  saveas(figure(6), save_file6, ext)
  saveas(figure(7), save_file7, ext)
  saveas(figure(8), save_file8, ext)
  
end  

