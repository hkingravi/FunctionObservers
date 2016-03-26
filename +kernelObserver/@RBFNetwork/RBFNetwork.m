%============================= RBFNetwork =================================
%
%  This class creates an instance of a linear rbf model, which can be 
%  trained in one of two ways:
%    a) Passing in the weights directly
%    b) Learning the weights from data and values in batch
%  The scaling parameter for the kernel can be passed in or can be inferred
%  using different methodologies. It's assumed that the scaling parameter
%  is shared across kernels, although future version will consider fitting
%  parameters to each kernel. 
%
%  Reference(s):   
%    -Chris Bishop -Pattern Recognition and Machine Learning
%                   1st Edition, Chapter 3.3, pgs 152-161. 
%                   Similar notation as the book is used. 
% 
%  Inputs:
%    centers    - dim x ncent center matrix, with each column as a center. 
%    k_func     - select kernel method: currently have Gaussian, Laplacian
%                 and Cauchy kernels
%    parameters - scalar scaling parameter common for radial kernels. 
%    noise      - standard deviation for noise
%    optimizer  - string indicating how to optimize hyperparameters: 
%                 choose from {'none', 'cv'} (optional)
%
%  Outputs:
%               -see functions
%
%============================= RBFNetwork =================================
%
%  Name:      RBFNetwork.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2014/06/03
%  Modified:  2014/06/03
%
%============================= RBFNetwork =================================
classdef RBFNetwork < handle 
  % class properties   
  properties (Access = public)    
    % some of these values are asked by the user in the constructor
    warnings   = 'on';
    debug_mode = 'on';   
  end
    
  % hidden variables 
  properties (Access = protected)
    k_obj               = [];   % kernel object, empty at initialization
    noise               = 0.1;  % current noise parameter
    ncent               = [];      
    k_params            = [];
    noise_params        = [];
    centers             = [];   % centers for RBF network; dim x ncent
    weights             = [];   % weights for centers; 1 x ncent  
    mapper              = [];   % generic finite-dimensional feature map
    optimizer           = [];
    jitter              = 1e-4;  
    validation_portion  = 0.1; 
    nparams             = 2;    % this is fixed for this class
    params_final        = [];   % used by FeatureSpaceGenerator objects
  end
  
  % class methods 
  methods
    
    function obj = RBFNetwork(centers_in, k_func_in, parameters_in, ...
                              noise_in, optimizer)
      %  Constructor for RBFNetwork.
      %
      %  Inputs:
      %    centers    - dim x ncent center matrix, with each column as a center.
      %    k_func     - select kernel method: {'gaussian','laplacian','cauchy'}
      %    parameters - scalar scaling parameter common for radial kernels.
      %    noise      - standard deviation for noise
      %    optimizer  - (optional) struct used for optimizing hyperparameters:
      %                 must have fields - method -{'none', 'cv', 'likelihood'}
      %                                  - nfolds - number of folds for cv
      %                                  - options parameters for minFunc
      %                                    for likelihood option
      %                                  - solver = {'primal', 'dual'} 
      %                                    for likelihood option
      %
      %  Outputs:
      %    -none
      % nesting structure to set new values 
      if nargin >=3
        addpath('../../minFunc/minFunc/')
        
        % parse centers
        if ~isnumeric(centers_in)
          exception = MException('VerifyInput:IncorrectType', ...
                                 'Centers must be in matrix form');
          throw(exception);
        else
          obj.centers = centers_in;
          obj.ncent   = size(centers_in,2);
          obj.weights = randn(obj.ncent,1); % initialize random weights
        end
                
        if nargin >=4
          obj.noise_params = noise_in; 
          
          if ~isnumeric(noise_in) || size(noise_in, 1) ~= 1
            exception = MException('VerifyInput:OutOfBounds', ...
                                    'Noise parameter must be positive');
            throw(exception);
          end          
          
          if nargin >=5
            if ~strcmp(optimizer.method, 'none') && ...
                ~strcmp(optimizer.method, 'cv') && ...
                ~strcmp(optimizer.method, 'likelihood')
              exception = MException('VerifyInput:OutOfBounds', ...
                                    ' incorrect choice of optimizer');
              throw(exception);
            end  
            obj.optimizer = optimizer;
          else
            obj.optimizer = struct('method', 'none');
          end            
          
          % finally, set noise value appropriately
          if strcmp(obj.optimizer.method, 'none') && size(noise_in, 2) > 1
            if strcmp(obj.warnings, 'on')
              disp(['RBFNetwork:WARNING: multidimensional noise passed in:' ...
                ' using first element of array'])
            end
          end  
          obj.noise = noise_in(1, 1); 
          if obj.noise < 0
            if strcmp(obj.warnings, 'on')
              disp('RBFNetwork:WARNING: negative noise passed in: setting to zero')
            end
            obj.noise = 0;
          end
        end 
        
        % parse kernel input, and initialize new kernelObj
        obj.k_params = parameters_in;
        
        if strcmp(obj.optimizer.method, 'none') && size(obj.k_params, 2) > 1
          if strcmp(obj.warnings, 'on')
            disp(['RBFNetwork:WARNING: multiple parameters for kernel' ...
              ' passed in: setting to first element of array'])
          end                              
        end
        obj.k_obj = kernelObserver.kernelObj(k_func_in, parameters_in(1));
        
      end

    end  
          
    function fit(obj,data, obs)
      %  Given new input data and observations , update the weight vector
      %  in batch form (i.e. using the normal equations).
      %
      %  Inputs:
      %    data  - dim x nsamp data matrix, passed in columnwise
      %    obs   - dim x nsamp observation matrix, passed in columnwise
      %
      %  Outputs:
      %    -none 
      if strcmp(obj.optimizer.method, 'none')        
        [weights_out, ~] = obj.fit_current(data, obs);
        obj.weights = weights_out;
      elseif strcmp(obj.optimizer.method, 'cv')
        % in this case, go over the existing parameters and compute
        % values
        nsamp = size(data, 2);
        nbands = size(obj.k_params, 2);
        nnoise = size(obj.noise_params, 2);
        cv_results = zeros(nbands, nnoise);
        cv_folds = cvpartition(nsamp, 'KFold', obj.optimizer.nfolds);
        
        for i=1:nbands
          for j=1:nnoise
            obj.k_obj.k_params = obj.k_params(i);
            obj.noise = obj.noise_params(j); 
            errs = zeros(1, cv_folds.NumTestSets);
            
            if strcmp(obj.debug_mode, 'on')
              disp(['CV on kernel params: ' num2str(obj.k_obj.k_params) ...
                    ', noise params: ' num2str(obj.noise)])
            end  
            
            for k=1:cv_folds.NumTestSets
              tr_idx = cv_folds.training(k);
              te_idx = cv_folds.test(k);
              
              xtrain = data(:, tr_idx);
              ytrain = obs(:, tr_idx);
              xtest = data(:, te_idx);
              ytest = obs(:, te_idx);              
              ypreds = obj.fit_and_predict(xtrain, ytrain, xtest);
              errs(i) = sum((ypreds-ytest).^2)/size(ypreds, 2);
            end  
            
            cv_results(i, j) = mean(errs);
          end
        end
        
        [band_ind, noise_ind] = kernelObserver.find_min_inds(cv_results);
        obj.k_obj.k_params = obj.k_params(band_ind);
        obj.noise = obj.noise_params(noise_ind);
        [weights_out, ~] = obj.fit_current(data, obs);
        obj.weights = weights_out;
        if strcmp(obj.debug_mode, 'on')
          disp(['Final kernel params: ' num2str(obj.k_obj.k_params) ...
                ', noise params: ' num2str(obj.noise)])
        end
      elseif strcmp(obj.optimizer.method, 'likelihood')
        % set up minFunc's parameters
        params = [log(obj.k_obj.k_params); log(obj.noise)];
        
        % parse options
        if ~isfield(obj.optimizer, 'useMex')
          options.useMex = 0;
        else          
          options.useMex = obj.optimizer.useMex;
        end  
        if ~isfield(obj.optimizer, 'Display')
          obj.optimizer.Display = 'off';
          options.Display = 'off';
        else
          options.Display = obj.optimizer.Display;
        end  
        if ~isfield(obj.optimizer, 'solver')
          obj.optimizer.solver = 'primal';        
        end  
        
        if ~strcmp(obj.optimizer.solver, 'primal') && ...
           ~strcmp(obj.optimizer.solver, 'dual')
         disp('Incorrect choice for optimizer.solver: resorting to primal')
         obj.optimizer.solver = 'primal';
        end
        
        options.MaxIter = 350;      
        disp(num2str(params))
        opt_params = minFunc(@obj.negloglik, params, options, data, obs);
        if strcmp(obj.debug_mode, 'on')
          fprintf('band = %.4f, noise = %.4f\n', exp(opt_params(1)),...
                                                 exp(opt_params(2)));
        end
        obj.k_obj.k_params = exp(opt_params(1));
        obj.noise = exp(opt_params(2));
        [weights_out, ~] = obj.fit_current(data, obs);
        obj.weights = weights_out;
        if strcmp(obj.debug_mode, 'on')
          disp(['Final kernel params: ' num2str(obj.k_obj.k_params) ...
                ', noise params: ' num2str(obj.noise)])
        end
      end
      
      obj.params_final = [obj.k_obj.k_params; obj.noise];
      obj.mapper = obj.create_map(); 
    end
    
    function [weights, mapped_data] = fit_current(obj, data, obs)
      %  Given new data, update the weight vector using least squares using
      %  current hyperparameters. 
      %
      %  Inputs:
      %    data  - dim x nsamp data matrix, passed in columnwise
      %    obs   - dim x nsamp observation matrix, passed in columnwise
      %
      %  Outputs:
      %    -none 
      obj.mapper = obj.create_map(); % create feature map using centers and kernel
      mapped_data = transpose(obj.mapper.transform(data)); % compute kernel matrix
      weights = (transpose(mapped_data)*mapped_data + ...
                 obj.noise^2*eye(obj.ncent))\(transpose(mapped_data)*transpose(obs));
    end  
    
    function [pred_te] = fit_and_predict(obj, data_tr, obs_tr, data_te)
      %  Method used for cross-validation. 
      %
      %  Inputs:
      %    data  - dim x nsamp data matrix, passed in columnwise
      %    obs   - dim x nsamp observation matrix, passed in columnwise
      %
      %  Outputs:
      %    -none 
      curr_weights = obj.fit_current(data_tr, obs_tr);
      pred_te = obj.predict(data_te, curr_weights);
    end      

    function K = transform(obj,data)
      %  Given new data, map it to feature space induced by kernel and
      %  centers. 
      %
      %  Inputs:
      %    data  - dim x nsamp data matrix, passed in columnwise
      %
      %  Outputs:
      %    K   - nsamp x ncent kernel matrix 
      K = obj.mapper.transform(data);
    end    
        
    function [f, K] = predict(obj, data_test, weights_in)
      %  Given a matrix data_test, compute the predictive values at each 
      %  point.
      %
      %  Inputs:
      %    data_test - dim x ntest data matrix, passed in columnwise
      %
      %  Outputs:
      %    -none 
      K = obj.mapper.transform(data_test);
      if nargin > 2
        f = transpose(weights_in)*K;
      else
        f = transpose(obj.weights)*K;
      end
    end    
    
    function [lik_val, param_deriv] = negloglik(obj, param_vec, data, obs)
      %  Given new input data and observations, compute the negative
      %  log-likelihood of the data, and return derivatives with respect to
      %  param_vec. Note that the input must be the variables in negative
      %  log form, to avoid them going negative. 
      %
      %  Inputs:
      %    param_vec - 2 x 1 parameter vector: [log(bandwidth); 
      %                                         log(noise)]      
      %    data  - dim x nsamp data matrix, passed in columnwise
      %    obs   - 1 x nsamp observation matrix, passed in columnwise
      %
      %  Outputs:
      %    lik_val - value for negative log likelihood. 
      %    param_vec - 2 x 1 parameter vector derivative: [bandwidth; noise]
      param_vec = exp(param_vec);
      obj.k_obj.k_params = param_vec(1);
      obj.noise = param_vec(2);
      curr_mapper = obj.create_map();
      
      if strcmp(obj.debug_mode, 'on')
          disp(['Current kernel params: ' num2str(obj.k_obj.k_params) ...
                ', noise params: ' num2str(obj.noise)])
      end
      
      % precompute useful constants and matrices
      nsamp = size(data, 2);      
      m_data = curr_mapper.transform(data);
      m_data_deriv = curr_mapper.get_deriv(data);      
            
      if strcmp(obj.optimizer.solver, 'dual')
        Amat = obj.noise^2*eye(nsamp) + transpose(m_data)*m_data; 
        Amat_deriv_band = transpose(m_data_deriv)*m_data + ...
                          transpose(m_data)*m_data_deriv;
        Amat_deriv_noise = 2*obj.noise^2*eye(nsamp);
        Amat_obs = Amat\transpose(obs);
        
        L = chol(Amat);
        logdetAmat = 2*sum(log(diag(L)));

        % compute negative log-likelihood
        lik_T1 = nsamp*log(2*pi);
        lik_T2 = logdetAmat;
        lik_val = 0.5*(lik_T1 + lik_T2 + obs*(Amat_obs));
        
        % compute derivatives
        band_deriv_T1 = trace(Amat\Amat_deriv_band);
        band_deriv_T2 = -obs*((Amat\Amat_deriv_band)*Amat_obs);
        band_deriv = 0.5*(band_deriv_T1 + band_deriv_T2);
        
        noise_deriv_T1 = trace(Amat\Amat_deriv_noise);
        noise_deriv_T2 = -obs*(Amat\Amat_obs);
        noise_deriv = 0.5*(noise_deriv_T1 + noise_deriv_T2);
        
      else
        Amat = eye(obj.ncent) + obj.ncent/(obj.ncent*(obj.noise^2))*m_data*transpose(m_data);
        m_data_p = Amat\m_data;
        m_data_d = Amat\m_data_deriv;
        Amat_deriv_band = obj.ncent/(obj.ncent*(obj.noise^2))*(m_data_deriv*transpose(m_data) + ...
          m_data*transpose(m_data_deriv));
        Amat_deriv_noise = -(2*obj.ncent)/(obj.noise^3*obj.ncent)*m_data*transpose(m_data);
        
        PhiMat = obj.ncent/(obj.ncent)*transpose(m_data)*m_data_p;
        m_data_dAmat = Amat\Amat_deriv_band;
        
        % compute negative log-likelihood
        L = chol(Amat);
        logdetAmat = 2*sum(log(diag(L)));
        
        lik_T1 = nsamp*log(2*pi);
        lik_T2 = nsamp*log(obj.noise^2) + logdetAmat; 
        T3_mat = 1/(obj.noise^2)*eye(nsamp) -...
          obj.ncent/(obj.noise^4*obj.ncent)*transpose(m_data)*m_data_p;
        lik_val = 0.5*(lik_T1 + lik_T2 + obs*T3_mat*transpose(obs));
        
        % term 1 w.r.t. bandwidth (eq.s {} in derivation)
        T1_band = 0.5*(trace(Amat\Amat_deriv_band));
        
        % term 2 w.r.t. bandwidth (eq.s {} in derivation)
        T2_band_B = (obj.ncent/obj.ncent)*transpose(m_data_deriv)*m_data_p;
        T2_band_Bp1 = (obj.ncent/obj.ncent)*m_data_d;
        T2_band_Bp2 = m_data_dAmat*m_data_p;
        T2_band_Bp = transpose(m_data)*(T2_band_Bp2 - T2_band_Bp1);
        T2_band = -1/(2*obj.noise^4)*obs*(T2_band_B + T2_band_Bp)*transpose(obs);
        band_deriv = T1_band + T2_band;
        
        % term 1 w.r.t. noise (eq.s {} in derivation)
        T1_noise = (nsamp/obj.noise) + ...
          0.5*trace(Amat\Amat_deriv_noise);
        T2_noise = -0.5*obs*((2/obj.noise^3)*eye(nsamp) + ...
          (2/obj.noise^7)*PhiMat*PhiMat - ...
          (4/obj.noise^5)*PhiMat)*transpose(obs);
        noise_deriv = T1_noise + T2_noise;
      end
      
      if strcmp(obj.optimizer.Display, 'on')
          disp(['Negative log-likelihood: ' num2str(lik_val) ...
                ', Kernel derivative params: band: ' num2str(band_deriv) ...
                ', noise: ' num2str(noise_deriv)])
      end
      
      param_deriv = [band_deriv; noise_deriv];
    end  
    
    function [mapper] = create_map(obj)
      %  Given the current kernel object and basis centers, construct a
      %  FeatureMap object and store it as the internal variable 'mapper'.
      %
      %  Inputs:
      %    -none
      %
      %  Outputs:
      %    -none       
      mapper = kernelObserver.FeatureMap('RBFNetwork');
      map_struct.centers = obj.centers; map_struct.kernel_obj = obj.k_obj;
      mapper.fit(map_struct);
    end  
    
    function params = get_params(obj)
      %  Return the kernel and observation noise parameters as one vector.
      %
      %  Inputs:
      %    -none
      %
      %  Outputs:
      %    -params - [kernel_parameters, noise]
      params = obj.params_final;
    end  
    
    function mval = get(obj,mfield)
      % Get a requested member variable.
      %      
      switch(mfield)
        case {'centers'}
          mval = obj.centers;
        case {'weights'}
          mval = obj.weights;
        case {'nparams'}
          mval = obj.nparams;
        case {'k_obj'}
          mval = obj.k_obj;
        case {'noise'}
          mval = obj.noise;          
        otherwise          
           disp('wrong variable name')
      end      
    end  
    
    function set(obj,mfield,mval)
      %
      %  Set a requested member variable.
      %      
      switch(mfield)
        case {'centers'}
          obj.centers = mval;
        case {'weights'}
          obj.weights = mval;
        otherwise          
           disp('wrong variable name')
      end        
  end
    
    
  end

end