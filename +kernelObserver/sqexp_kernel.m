%============================ sqexp_kernel ================================
%  
%  This code takes as input two data matrices, and returns a kernel matrix
%  evaluated between the points in the matrices computing the squared
%  exponential kernel k(x,y) = sf^2e^{-(x-y)^TC^{-1}(x-y)}, where 
%  C is diagonal with parameters ell_1^2,...,ell_D^2, where
%  D is the dimension of the input space and sf2 is the signal variance.
%  Based on GPML toolbox. 
%
%  Reference(s): 
% 
%  INPUT:
%    data1	    - d x n data matrix, with each column as an observation. 
%    data2	    - d x m data matrix, with each column as an observation.
%    bandwidth  - scaling parameter for RBF kernel. 
%
%  OUTPUT:
%               - n x m kernel matrix 
%
%============================ sqexp_kernel ================================
%
%  Name:        sqexp_kernel.m
%
%  Author:      Hassan A. Kingravi
%
%  Created:  	2016/03/30
%  Modified: 	2016/03/30
%
%============================ sqexp_kernel ================================
function [Kmat, deriv_cell] =  sqexp_kernel(x, y, params)
ndim = size(x, 1);
ell = params(1:ndim);  
sf2 = params(ndim+1)^2;
Kmat = kernelObserver.dist_mat(diag(1./ell)*x,diag(1./ell)*y);
Kmat = sf2*exp(-Kmat/2);   

if nargout > 1
  nderivs = length(params)-1;
  deriv_cell = cell(1, nderivs+1);
  for i=1:nderivs
    deriv_cell{i} = Kmat.*kernelObserver.dist_mat(x(i, :)/ell(i),...
                                                  y(i, :)/ell(i));
  end  
  deriv_cell{nderivs+1} = 2*Kmat; 
end  

end
