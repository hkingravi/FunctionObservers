%====================== negative_log_likelihood ===========================
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
%    param_vec   - (dim+1) x 1 hyperparameter vector, with dim 
%                  parameters for the kernel, and the last being 
%                  the observation noise parameter
%    basis       - d x m data matrix, with each column as an basis vector.
%    data  	     - d x n data matrix, with each column as a data location.
%    obs         - 1 x n observation matrix
%    k_type      - kernel type for feature map
%    mapper_type - type of feature map
%    solver_type - choose from ['primal', 'dual']
%    
%
%  OUTPUT:
%               - n x m kernel matrix 
%
%====================== negative_log_likelihood ===========================
%
%  Name:        negative_log_likelihood.m
%
%  Author:      Hassan A. Kingravi
%
%  Created:  	2014/03/27
%  Modified: 	2016/03/30
%
%====================== negative_log_likelihood ===========================
function [nll, derivs] =  negative_log_likelihood_rks(param_vec, nbases, ndim, seed, ...
                                                      data, obs, k_type, ...
                                                      mapper_type, solver_type)

if ~strcmp(solver_type, 'primal') && ~strcmp(solver_type, 'dual')
  exception = MException('VerifyInput:OutOfBounds', ...
                         ' incorrect choice of solver: choose from',...
                         ' primal or dual');
  throw(exception);
end  
                                                
% compute kernel map, and then kernel matrix 
nsamp = size(data, 2);
dim = length(param_vec);
k_params = exp(param_vec(1:dim-1)); 
noise = exp(2*param_vec(dim));  % do this to avoid negative parameter scaling issues
k_obj = kernelObserver.kernelObj(k_type, k_params);
jitter = 1e-7;

mapper = kernelObserver.FeatureMap(mapper_type);
map_struct.kernel_obj = k_obj;
map_struct.nbases = nbases; 
map_struct.ndim = ndim;
map_struct.seed = seed;
mapper.fit(map_struct);
m_data = mapper.transform(data);

nrbases = 2*nbases;  % RKS bases is twice the number due to complex domain

if strcmp(solver_type, 'dual')
  K = transpose(m_data)*m_data;

  % missing bits about what happens for small noise parameters
  if noise < 1e-6
    L = chol(K + (noise + jitter)*eye(nsamp)); sl = 1;
  else
    L = chol(K/noise + eye(nsamp)); sl = noise;
  end  
  
  alpha = kernelObserver.solve_chol(L, obs')/sl;
  Logdet = sum(log(diag(L)));
  Cinv = kernelObserver.solve_chol(L, eye(nsamp))/sl; 
else
  % in this case, the primal must be solved. First, compute the capacitance
  % matrix
  Kp = m_data*transpose(m_data);
  
  % missing bits about what happens for small noise parameters
  if noise < 1e-6
    L = chol(Kp + (noise + jitter)*eye(nrbases)); sl = 1;
  else
    L = chol(Kp/noise + eye(nrbases)); sl = noise; 
  end  
  
  Ainv = kernelObserver.solve_chol(L, eye(nrbases));
  Cinv = (eye(nsamp) - (transpose(m_data)*Ainv*m_data)/sl)/sl;
  alpha = Cinv*transpose(obs);
  Logdet = sum(log(diag(L)));
end

if nargout >= 1
  nll = obs*alpha/2 + Logdet + nsamp*log(2*pi*sl)/2;   % compute negative log-likelihood
  if nargout >= 2
    % need to compute derivatives    
    md_data = mapper.get_deriv(data);
    nderivs = size(md_data, 2);
    derivs = zeros(nderivs+1, 1);
    Q = Cinv - alpha*alpha';     % precompute for convenience
    for i=1:nderivs
      bandMat = transpose(md_data{i})*m_data + transpose(m_data)*md_data{i};
      derivs(i) = sum(sum(Q.*bandMat))/2;
    end          
    noiseMat = 2*noise*eye(nsamp);
    derivs(nderivs+1) = sum(sum(Q.*noiseMat))/2;
  end
end


end
