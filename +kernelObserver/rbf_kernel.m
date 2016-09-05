%============================= rbf_kernel =================================
%  
%  This code takes as input two data matrices, and returns a kernel matrix
%  evaluated between the points in the matrices. This kernel is different
%  from the standard 'kernel' function, as it's designed for the RBFNetwork
%  class, which has more general kernels. This function requires 4
%  arguments, in which the kernel type must be included. For the time
%  being, only pass in a multiscale bandwidth vector for the Gaussian;
%  otherwise, the result returned will NOT be correct. 
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
%    bandwidth  - scaling parameter for RBF kernel. 
%
%  OUTPUT:
%               - n x m kernel matrix 
%
%============================= rbf_kernel =================================
%
%  Name:        rbf_kernel.m
%
%  Author:      Hassan A. Kingravi
%
%  Created:  	2014/06/03
%  Modified: 	2016/03/14
%
%============================= rbf_kernel =================================
function [Kmat, deriv_cell] =  rbf_kernel(x, y, k_func, bandwidth)

% compute distances of points to each other
if(length(bandwidth) ~= 1) % scalar bandwidth
  disp(['rbf_kernel:WARNING: multidimensional bandwidth passed in:' ...
       ' using first element of array'])
  bandwidth = bandwidth(1, 1);
end

Dmat = kernelObserver.dist_mat(x, y);

if strcmpi(k_func,'laplacian')
  Dmat = sqrt(Dmat);
  Kmat = exp(-Dmat./(bandwidth));
elseif strcmpi(k_func,'cauchy')
  Kmat = 1./(1 + Dmat./bandwidth);
elseif strcmpi(k_func,'gaussian')
  Kmat = exp(-Dmat./(2*bandwidth^2));  
end

if nargout > 1
  % in this case, return a derivative
  nderivs = 1;
  deriv_cell = cell(1, nderivs);
  if strcmpi(k_func,'laplacian')
    % need to implement this
    except = MException('VerifyInput:OutOfBounds', ...
                        ' derivative for Laplacian',...
                        ' kernel currently not implemented');
    throw(except);
  elseif strcmpi(k_func,'cauchy')
    % need to implement this
    except = MException('VerifyInput:OutOfBounds', ...
                        ' derivative for Cauchy',...
                        ' kernel currently not implemented');
    throw(except);
  elseif strcmpi(k_func,'gaussian')    
    deriv_cell{1} = (1/(bandwidth^3))*Kmat.*Dmat;             
  end
end  

end
