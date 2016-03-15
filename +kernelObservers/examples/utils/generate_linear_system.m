%======================== generate_linear_system ==========================
%  
%  This function generates a generic linear system, with as many
%  oscillations as possible. The goal is to get an A matrix that generates
%  interesting behavior, and to use that matrix for the weights of an RBF
%  network. This is done using random behavior generating the poles. It's
%  assumed that the dimensionality of the system is even; this is not
%  necessary in general, but makes the placement a little easier in this
%  case, due to 
% 
%  Reference(s): 
%    none
% 
%  Inputs:
%    input_dim  - 1 x 1 scalar specifying dimension of the system 
%    Ts         - 1 x nsamp vector of sampling times 
%    seed       - 1 x 1 scalar specifying random seed
%    sys_type   - {'continuous','discrete'}
%    
%  Outputs:
%    A          - input_dim x input_dim matrix 
%
%======================== generate_linear_system ==========================
%  Name:  generate_linear_system.m
%
%  Author(s): Hassan A. Kingravi
%
%  Created:  2014/05/14
%  Modified: 2014/05/21
%======================== generate_linear_system ==========================
function [ A,B,C ] = generate_linear_system(input_dim, seed, sys_type, inputs)

% set the random seed 
s = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(s);

if ~strcmp(sys_type,'continuous') && ~strcmp(sys_type,'discrete')
  disp('Incorrect system type: using continuous time.')
  sys_type = 'continuous';
end

% decide on percentage of oscillating modes and non-oscillating modes 
osc = 0.5;
marg_osc = 0.4;

if nargin < 4
  inputs = 12;
end

plot_on = 0; 

A = randn(input_dim);
B = randn(input_dim,inputs);
%B(end) = 1; 
C = eye(input_dim);
pole_matrix = zeros(1,input_dim);

num_osc_pole = (osc*input_dim)/2;
pole_counter = 1; 
% generate poles; since this is a discrete system, absolute value of the
% poles must be strictly inside the unit ball 
while pole_counter < num_osc_pole
  
  if strcmp(sys_type,'discrete')
  
    real_pole_val = rand; % generate random pole
    img_pole_val = rand;
    
    marg_test = rand; % sometimes, you need to have a pole that's marginally stable    
    
    z = real_pole_val-img_pole_val*1i;
    
    if abs(z) > 1 || marg_test < 0.5
      z = z/(abs(z));
    end
    
  else
    marg_test = rand % sometimes, you need to have a pole that's marginally stable
    
    if marg_test < marg_osc
      real_pole_val = 0
    end
    
    real_pole_val = -abs(randn); % generate random pole
    img_pole_val = 1;
    
    z = real_pole_val-img_pole_val*1i;
  
  end

  % place complex poles 
  pole_matrix(pole_counter) = z;
  pole_counter = pole_counter + 1; 
  pole_matrix(pole_counter) = conj(z);    
  pole_counter = pole_counter + 1; 
end

% now, place poles for decaying terms, for extra complexity 
while pole_counter <= input_dim
  
  if strcmp(sys_type,'discrete')
    real_pole_val = rand; % generate random pole
  else
    real_pole_val = -abs(randn); % generate random pole    
  end  

  pole_matrix(pole_counter) = real_pole_val; 
  pole_counter = pole_counter + 1; 
end


% once all this stuff is done, place poles, and compute A matrix 
P = place(A,B,pole_matrix); 
A = A - B*P; 

% plot system response 
if plot_on == 1
  eig(A)
  sys=ss(A,B,C,0);
  initial(sys,randn(input_dim,1))
end

end

