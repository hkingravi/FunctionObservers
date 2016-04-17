%===================== negative_log_likelihood_gp =========================
%  
%  This code computes the negative log-likelihood of an Gaussian process
%  model. The code is heavily based on the GPML toolbox by Rasmussen et al. 
%
%  Reference(s): Rasmussen and Williams: "Gaussian Processes for Machine 
%                                         Learning".
%                Rasmussed et al: GPML Toolbox. 
%                
% 
%  INPUT:
%    param_vec   - (dim+1) x 1 hyperparameter vector, with dim 
%                  parameters for the kernel, and the last being 
%                  the observation noise parameter
%    basis       - d x m data matrix, with each column as an basis vector.
%    data  	     - d x n data matrix, with each column as a data location.
%    obs         - 1 x n observation matrix
%    k_type      - kernel type for feature map
%
%  OUTPUT:
%               - n x m kernel matrix 
%
%===================== negative_log_likelihood_gp =========================
%
%  Name:        negative_log_likelihood.m
%
%  Author:      Hassan A. Kingravi
%
%  Created:  	2014/03/27
%  Modified: 	2016/03/30
%
%===================== negative_log_likelihood_gp =========================
function [nll, derivs] =  negative_log_likelihood_gp(param_vec, data, ...
                                                     obs, k_type)

% compute kernel map, and then kernel matrix 
nsamp = size(data, 2);
dim = length(param_vec);
k_params = exp(param_vec(1:dim-1)); 
noise = exp(2*param_vec(dim));  % do this to avoid negative parameter scaling issues
k_obj = kernelObserver.kernelObj(k_type, k_params);
jitter = 1e-7;

[K, deriv_cell] = kernelObserver.generic_kernel(data, data, k_obj);

% missing bits about what happens for small noise parameters
if noise < 1e-6
  L = chol(K + (noise + jitter)*eye(nsamp)); sl = 1;
else
  L = chol(K/noise + eye(nsamp)); sl = noise;
end

alpha = kernelObserver.solve_chol(L, obs')/sl;
Logdet = sum(log(diag(L)));
Cinv = kernelObserver.solve_chol(L, eye(nsamp))/sl;

if nargout >= 1
  nll = obs*alpha/2 + Logdet + nsamp*log(2*pi*sl)/2;   % compute negative log-likelihood
  if nargout >= 2
    % need to compute derivatives    
    nderivs = length(deriv_cell);
    derivs = zeros(nderivs+1, 1);
    Q = Cinv - alpha*alpha';     % precompute for convenience
    for i=1:nderivs
      bandMat = deriv_cell{i};
      derivs(i) = sum(sum(Q.*bandMat))/2;
    end          
    noiseMat = 2*noise*eye(nsamp);
    derivs(nderivs+1) = sum(sum(Q.*noiseMat))/2;
  end
end


end
