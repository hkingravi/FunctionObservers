%============================== lik_dual ==================================
%  
%  This code the negative log-likelihood of an approximate Gaussian process
%  model in its dual form. The code is heavily based on the GPML toolbox by
%  Rasmussen et al. 
%
%  Reference(s): Rasmussen and Williams: "Gaussian Processes for Machine 
%                                         Learning".
%                Rasmussed et al: GPML Toolbox. 
%                
% 
%  INPUT:
%    basis      - d x m data matrix, with each column as an basis vector.
%    data  	    - d x n data matrix, with each column as a data location.
%    obs        - 1 x n observation matrix
%    k_obj      - kernel object
%    params     - (dim+1) x 1 hyperparameter vector, with dim 
%                 parameters for the kernel, and the last being 
%                 the observation parameter for the data
%
%  OUTPUT:
%               - n x m kernel matrix 
%
%============================== lik_dual ==================================
%
%  Name:        lik_dual.m
%
%  Author:      Hassan A. Kingravi
%
%  Created:  	2014/03/27
%  Modified: 	2016/03/27
%
%============================== lik_dual ==================================
function [nll, derivs] =  lik_dual(param_vec, basis, data, obs, k_type, mapper_type)

% compute kernel map, and then kernel matrix 
nsamp = size(data, 2);
dim = length(param_vec);
k_params = param_vec(1:dim-1); 
noise = exp(2*param_vec(dim));  % do this to avoid negative parameter scaling issues
k_obj = kernelObserver.kernelObj(k_type, k_params);

mapper = kernelObserver.FeatureMap(mapper_type);
map_struct.kernel_obj = k_obj;
map_struct.centers = basis; 
mapper.fit(map_struct);
m_data = mapper.transform(data);
K = transpose(m_data)*m_data; 

L = chol(K/noise + eye(nsamp)); sl = noise; 
alpha = kernelObserver.solve_chol(L, obs')/sl;

if nargout >= 1  
  nll = obs*alpha/2 + sum(log(diag(L))) + nsamp*log(2*pi*sl)/2;   % compute negative log-likelihood
  if nargout >= 2
    % need to compute derivatives
    derivs = zeros(2, 1);
    md_data = mapper.get_deriv(data);
    Q = kernelObserver.solve_chol(L, eye(nsamp))/sl - alpha*alpha';     % precompute for convenience
    bandMat = transpose(md_data)*m_data + transpose(m_data)*md_data;
    noiseMat = 2*noise*eye(nsamp);
    derivs(1) = sum(sum(Q.*bandMat))/2;
    derivs(2) = sum(sum(Q.*noiseMat))/2;
  end
end  

end
