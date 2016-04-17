%============================ generic_kernel ==============================
%  
%  A generic kernel function called by classes which can compute different
%  kinds of kernels. 
%
%  Reference(s): 
% 
%  INPUT:
%    data1	    - d x n data matrix, with each column as an observation. 
%    data2	    - d x m data matrix, with each column as an observation.
%    k_func     - string denoting kernel type: 
%                 - 'laplacian' 
%                 - 'cauchy'
%                 - any other string defaults to the Gaussian 
%                   (squared exponential)
%    parameters - parameters for the type of kernels you want. 
%
%  OUTPUT:
%               - n x m kernel matrix 
%
%=========================== generic_kernel ===============================
%
%  Name:        generic_kernel.m
%
%  Author:      Hassan A. Kingravi
%
%  Created:  	2014/06/03
%  Modified: 	2016/04/10
%
%=========================== generic_kernel ===============================
function [kmat, deriv_cell] =  generic_kernel(x, y, k_obj)

k_func = k_obj.k_name;
parameters = k_obj.k_params;
if strcmpi(k_func,'laplacian') || strcmpi(k_func,'cauchy')...
                               || strcmpi(k_func,'gaussian')
  bandwidth = parameters(1);  
  [kmat, deriv_cell] = kernelObserver.rbf_kernel(x, y, k_func, bandwidth);
elseif strcmpi(k_func,'periodic')    
  [kmat, deriv_cell] = kernelObserver.periodic_kernel(x, y, ...
                                                      parameters(1),...
                                                      parameters(2)); 
elseif strcmpi(k_func,'locally_periodic')    
  kmat1 = kernelObserver.periodic_kernel(x,y,parameters(1),parameters(2));   
  kmat2 = kernelObserver.rbf_kernel(x,y,'gaussian',parameters(2));   
  kmat = kmat1.*kmat2; 
elseif strcmpi(k_func,'spectral')
  % in this case, the 'parameter' is a Gaussian mixture model
  centers = parameters.mu;
  bandwidth = parameters.Sigma;
  weights = parameters.PComponents;
  
  K = zeros(size(x, 2), size(y, 2));
  for i=1:length(weights)
    % compute exponential part of kernel
    curr_bandwidth = 2*pi^2*bandwidth(1, :, i);
    d =  (x'*curr_bandwidth)*y;
    dx = sum((x'*curr_bandwidth)'.*x,1);
    dy = sum((y'*curr_bandwidth)'.*y,1);
    val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;    
    curr_K = exp(-val);            
    K = K + curr_K; 
  end
    
end

end
