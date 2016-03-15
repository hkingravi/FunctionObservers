%========================== periodic_kernel ===============================
%  
%  This code takes as input two data matrices, and returns a kernel matrix
%  evaluated between the points in the matrices featuring the periodic
%  kernel. Currently, it's assumed the data is 1D, and passed in
%  column-wise. 
%
%  Reference(s): 
% 
%  INPUT:
%    data1	    - 1 x n data matrix, with each column as an observation. 
%    data2	    - 1 x m data matrix, with each column as an observation.
%    period     - period for sinusoid portion of kernel. 
%    bandwidth  - scaling parameter for exponential portion of kernel. 
%
%  OUTPUT:
%               - n x m kernel matrix 
%
%========================== periodic_kernel ===============================
%
%  Name:        periodic_kernel.m
%
%  Author:      Hassan A. Kingravi
%
%  Created:  	2015/02/03
%  Modified: 	2015/02/03
%
%========================== periodic_kernel ===============================
function kmat =  periodic_kernel(x,y,period,bandwidth)

N = size(x,2);
M = size(y,2);

dx_vals = x; 
dy_vals = y; 
s_val = -2.0/(bandwidth^2); % compute scaling 

val1 = repmat(transpose(dx_vals), 1, M);
val2 = repmat(dy_vals, N, 1); 
kmat = sin((val1-val2)./period);
kmat = kmat.^2;
kmat = exp(s_val.*kmat);

end
