function [nll, derivs] =  inverse_check(param_vec, basis, ...
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
k_params = param_vec(1:dim-1); 
noise = exp(2*param_vec(dim));  % do this to avoid negative parameter scaling issues
k_obj = kernelObserver.kernelObj(k_type, k_params);

mapper = kernelObserver.FeatureMap(mapper_type);
map_struct.kernel_obj = k_obj;
map_struct.centers = basis; 
mapper.fit(map_struct);
m_data = mapper.transform(data);


K = transpose(m_data)*m_data;

% missing bits about what happens for small noise parameters
L = chol(K/noise + eye(nsamp)); sl = noise;
alpha = kernelObserver.solve_chol(L, obs')/sl;

if strcmp(solver_type, 'dual')
  
  
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
else
  % in this case, the primal must be solved. First, compute the capacitance
  % matrix
  ncent = size(basis, 2);
  Kd = m_data*transpose(m_data);
  
  % missing bits about what happens for small noise parameters
  L = chol(Kd/noise + eye(ncent)); sl = noise; 
  Ainv = kernelObserver.solve_chol(L, eye(ncent));
  
  % hack for log-likelihood
  
end


end
