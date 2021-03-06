%========================== kalman_filter_test ============================
%  
%  This code tests the KalmanFilter class. The goal is to measure the
%  position of an aircraft through a model for radar measurements. We use
%  a dynamical model for the position of the aircraft, utilizing states 
%  [x; x_dot; y; y_dot]. The measurement model just gets the current
%  position of the aircraft [x; y], but with noise in the measurements. 
%
%========================== kalman_filter_test ============================
%
%  Name:	kalman_filter_test.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  2016/05/02
%  Modified: 2016/05/02
%
%========================== kalman_filter_test ============================
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
f_lwidth = 3; 
f_marksize = 5;
c_marksize = 15;
font_size = 15;
save_results = 1; 
ext = 'png';

% set seed
seed = 20; 
s = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(s);
add_process_noise = 'yes';

% save data for unit test
save_unit_test = 0; 

%% filter parameters
Ac = [1 1 0 0; 0 1 0 0; 0 0 1 1; 0 0 0 1];  % dynamics model for radar 
C = [1 0 0 0; 0 0 1 0];

dyn_dim = size(Ac, 1);
meas_dim = size(C, 1);

dt = 0.001;  % time step
A = eye(size(Ac, 1)) + dt.*Ac;  % discretize dynamics operator

nmeas = size(C, 1);
ncent = size(A, 1);

P_init = 0.0001*eye(ncent);
Q = 0.0001*eye(ncent);
R = 0.1*eye(nmeas);
QL = chol(Q);
RL = chol(R);
m_init = [2; 0.001; -3; -0.01];
rand_init = randn(dyn_dim, 1);

%% initialize KalmanFilter, and compute filter measurements and corrections
kf = kernelObserver.KalmanFilter(P_init, Q, R);
kf.fit(A, C, rand_init);

time_steps = 1000; 
states_noisy = zeros(ncent, time_steps);
meas_noisy = zeros(nmeas, time_steps);
meas_actual = zeros(nmeas, time_steps);
meas_kalman = zeros(nmeas, time_steps);
curr_state = m_init; 

% generate measurements
for i=1:time_steps
  meas_noisy(:, i) = C*curr_state + RL*randn(nmeas, 1);
  meas_actual(:, i) = C*curr_state;
  
  pred_state = kf.predict(meas_noisy(:, i));
  meas_kalman(:, i) = C*pred_state;
  
  if strcmp(add_process_noise, 'yes')
    states_noisy(:, i) = curr_state + QL*randn(ncent, 1);
  else
    states_noisy(:, i) = curr_state;
  end
  curr_state = A*curr_state;
end

%% plot final results
figure(1);
plot(meas_noisy(1, :), 'ro')
hold on; 
plot(meas_actual(1, :), 'b-', 'LineWidth', f_lwidth)
hold on; 
plot(meas_kalman(1, :), 'g-', 'LineWidth', f_lwidth)
legend('measured', 'actual', 'kalman')
title('Measurements and recovery of x position of aircraft from radar measurements')
xlabel('Time steps')
ylabel('x position')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

figure(2);
plot(meas_noisy(2, :), 'ro')
hold on; 
plot(meas_actual(2, :), 'b-', 'LineWidth', f_lwidth)
hold on; 
plot(meas_kalman(2, :), 'g-', 'LineWidth', f_lwidth)
legend('measured', 'actual', 'kalman')
title('Measurements and recovery of y position of aircraft from radar measurements')
xlabel('Time steps')
ylabel('y position')
set(gca,'FontSize', font_size)
set(findall(gcf,'type','text'),'FontSize', font_size)

set(figure(1),'Position',[100 100 800 600]);
set(figure(2),'Position',[100 100 800 600]);

if save_results == 1
  save_file1 = './results/kalman_filter_fig1_xpos';
  save_file2 = './results/kalman_filter_fig2_ypos';
  
  set(figure(1),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  set(figure(2),'PaperOrientation','portrait','PaperSize', [8.5 7],...
    'PaperPositionMode', 'auto', 'PaperType','<custom>');
  saveas(figure(1), save_file1, ext)
  saveas(figure(2), save_file2, ext)
end

