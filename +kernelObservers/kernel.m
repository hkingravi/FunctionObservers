%=============================== kernel ===================================
%  
%  This code takes as input two data matrices, and returns a kernel matrix
%  evaluated between the points in the matrices. 
%
%  Reference(s): 
% 
%  INPUT:
%    data1	    - d x n data matrix, with each column as an observation. 
%    data2	    - d x m data matrix, with each column as an observation. 
%    sigma      - covariance parameters; currently, this is the sigma
%                 value for the Gaussian kernel function, (which Rasmussen 
%                 calls the 'squared exponential'). 
%  OUTPUT:
%               - n x m kernel matrix 
%
%=============================== kernel ===================================
%
%  Name:		kernel.m
%
%  Author: 		Hassan A. Kingravi
%
%  Created:  	2013/08/12
%  Modified: 	2013/08/12
%
%=============================== kernel ===================================
function v =  kernel(x,y,sigma,A)

if nargin == 3
  nu = 1; 
elseif nargin == 4
  nu = A; 
end

if(length(sigma) == 1) % isotropic covariance (the same across all dimensions)
  d = x'*y;
  dx = sum(x.^2,1);
  dy = sum(y.^2,1);
  val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
  v = nu.*exp(-val./(2*sigma^2));
else                   % anisotropic covariance
  isigma = inv(diag(sigma.^2));
  d =  (x'*isigma)*y;
  dx = sum((x'*isigma)'.*x,1);
  dy = sum((y'*isigma)'.*y,1);
  val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
  v = nu .* exp(-val./2);
end

end
