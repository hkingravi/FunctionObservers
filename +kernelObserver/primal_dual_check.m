function [nll_dual, nll_primal] =  primal_dual_check(param_vec, basis, ...
                                           data, obs, k_type, ...
                                           mapper_type, solver_type)
%primal_dual_check Check primal vs. dual formulations of kernel updates.
% This code gives us confidence that primal-dual versions
% of the kernel inverses are implemented correctly for both computing the
% negative log-likelihood, and the gradients with respect to the parameters.
%


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


%% primal section
K = transpose(m_data)*m_data;

% ensure noise isn't too small
if noise < 1e-6                       
  L = chol(K + noise*eye(nsamp)); sl = 1;
else
  L = chol(K/noise + eye(nsamp)); sl = noise; 
end
alpha_dual = kernelObserver.solve_chol(L, obs')/sl;
nll_dual = obs*alpha_dual/2 + sum(log(diag(L))) + nsamp*log(2*pi*sl)/2;   % compute negative log-likelihood
derivs = zeros(2, 1);
md_data = mapper.get_deriv(data);
Q = kernelObserver.solve_chol(L, eye(nsamp))/sl - alpha_dual*alpha_dual';     % precompute for convenience
bandMat = transpose(md_data{1})*m_data + transpose(m_data)*md_data{1};
noiseMat = 2*noise*eye(nsamp);
derivs(1) = sum(sum(Q.*bandMat))/2;
derivs(2) = sum(sum(Q.*noiseMat))/2;

Cinv_dual = kernelObserver.solve_chol(L, eye(nsamp))/sl;

%% primal section
% in this case, the primal must be solved. First, compute the capacitance
% matrix
ncent = size(basis, 2);
Kp = m_data*transpose(m_data);

% missing bits about what happens for small noise parameters
if noise < 1e-6   
  Lp = chol(Kp + noise*eye(ncent)); sl = 1;  % not correct. Need to examine
else
  Lp = chol(Kp/noise + eye(ncent)); sl = noise;
end
Ainv = kernelObserver.solve_chol(Lp, eye(ncent));
Cinv_primal = (eye(nsamp) - (transpose(m_data)*Ainv*m_data)/sl)/sl;

disp(['Norms of kernel inverse matrices in matrix norm (dual-primal): '...
       num2str(norm(Cinv_dual)) ', ' num2str(norm(Cinv_primal))])
disp(['Difference between kernel inverse matrices in matrix norm: '...
       num2str(norm(Cinv_dual-Cinv_primal))])

% now check outputs for alpha vectors 
alpha_primal = Cinv_primal*transpose(obs);
disp(['Difference between alpha vectors in vector norm: '...
       num2str(norm(alpha_dual-alpha_primal))])

% compare determinants 
Klogdet = sum(log(diag(L)));
Kplogdet = sum(log(diag(Lp)));
     
disp(['Log-determinants of kernel matrices (dual-primal): '...
       num2str(Klogdet) ', ' num2str(Kplogdet)])

% compute primal negative log-likelihood
nll_primal = obs*alpha_primal/2 + Kplogdet + nsamp*log(2*pi*sl)/2;   % compute negative log-likelihood

disp(['Negative log-likelihoods (dual-primal): '...
       num2str(nll_dual) ', ' num2str(nll_primal)])
disp(['Absolute difference between negative log-likelihoods: '...
       num2str(norm(nll_primal-nll_dual))])


end
