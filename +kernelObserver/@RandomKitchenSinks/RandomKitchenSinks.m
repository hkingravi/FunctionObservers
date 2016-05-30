%========================== RandomKitchenSinks ============================
%
%  This class creates an instance of an approximate feature map for the
%  Gaussian kernel. These are typically known as the Random Fourier 
%  Features, but Smola has been calling them RKSs for a while, and they may
%  be a special case. 
%
%  Reference(s):   
%    -Rahimi & Recht - Random Features for Large-Scale Kernel Machines
% 
%  Inputs:
%    sigma      - parameter for kernel
%    nbases      - number of bases: you will generate a map TWICE this
%                 length!
%    ndim        -dimensionality of data
%
%  Outputs:
%               -see functions
%
%========================== RandomKitchenSinks ============================
%
%  Name:      RandomKitchenSinks.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2015/10/01
%  Modified:  2015/10/01
%
%========================== RandomKitchenSinks ============================
classdef RandomKitchenSinks < handle 
  % class properties   
  properties (Access = public)        
    warnings   = 'on';
    debug_mode = 'on';       
  end
    
  % hidden variables 
  properties (Access = protected)    
    sigma               = 1; 
    noise               = 0.1;  % Regularization parameter     
    nbases              = [];   % number of actual bases
    nbases_rks          = [];   % number of RKS bases
    ndim                = [];
    nparams             = 2;    % this is fixed for this class        
    seed                = 0;    % need better way to set this    
    sort_mat            = 0; 
    k_obj               = [];
    weights             = [];
    mapper              = [];
    param_struct        = [];
    params_final        = [];   % used by FeatureSpaceGenerator objects    
  end
  
  % class methods 
  methods
    
    function obj = RandomKitchenSinks(nbases_rks, ndim, k_func_in, parameters_in, ...
                                      noise_in, param_struct_in)
      %  Constructor for RandomKitchenSinks.
      %
      %  Inputs:
      %    nbases_rks       - number of bases: you will generate a map 
      %                       TWICE this length!
      %    ndim             - dimensionality of data
      %    k_func_in        - 'gaussian' only for now
      %    parameters_in    - Gaussian kernel radial parameter (bandwidth)
      %    noise_in         - observation noise initialization
      %    param_struct_in  - struct containing extra parameters,
      %                       particularly for the min_func package for
      %                       optimizing the hyperparameters. Can have
      %                       fields
      %                       - 'method': what method to choose for
      %                                   optimization (none, which just 
      %                                   trains weights using the initial
      %                                   parameters, cross-validation,
      %                                   or likelihood maximization): 
      %                                   {'none', 'cv', 'likelihood'}
      %                       - 'useMex': use compiled C code in minFunc
      %                       - 'Display': display minFunc iterations
      %                       - 'solver': {'primal', 'dual'} for large
      %                                   datasets, use 'primal'
      %                       - 'DerivativeCheck': {'off', 'on'}
      %                       - 'sort_mat': only use for single-dimensional
      %                                     inputs, to get descriptive 
      %                                     pictures of the Fourier 
      %                                     operator: {0, 1} 
      %
      %  Outputs:
      %    -none      
      obj.nbases_rks = nbases_rks;
      obj.nbases = 2*nbases_rks;
      obj.ndim = ndim;
      obj.sigma = parameters_in(1);  % radial kernel parameter
      
      if ~strcmp(k_func_in, 'gaussian')
        exception = MException('VerifyInput:OutOfBounds', ...
          'Kernel function must be Gaussian');
        throw(exception);
      end
          
      obj.k_obj = kernelObserver.kernelObj(k_func_in, parameters_in(1));      
      
      if nargin > 4
          obj.noise = noise_in; 
          if nargin > 5
            if ~strcmp(param_struct_in.method, 'none') && ...
                ~strcmp(param_struct_in.method, 'cv') && ...
                ~strcmp(param_struct_in.method, 'likelihood')
              exception = MException('VerifyInput:OutOfBounds', ...
                ' incorrect choice of optimizer');
              throw(exception);
            end
            obj.param_struct = param_struct_in;
            if isfield(obj.param_struct, 'sort_mat')
              obj.sort_mat = obj.param_struct.sort_mat; 
            end
          end
      end    
      
    end  
          
    function fit(obj, data, obs)
      %  Given new data (possibly in batch form), update the weight vector
      %  in batch form (i.e. using the normal equations).
      %
      %  Inputs:
      %    data  - ndim x nsamp data matrix, passed in columnwise
      %    obs   - ndim x nsamp observation matrix, passed in columnwise
      %
      %  Outputs:
      %    -none 
      if strcmp(obj.param_struct.method, 'none')
        [weights_out, ~] = obj.fit_current(data, obs);
        obj.weights = weights_out;
      elseif strcmp(obj.param_struct.method, 'cv')
      
      elseif strcmp(obj.param_struct.method, 'likelihood')
        addpath('../../minFunc/minFunc/')
        addpath('../../minFunc/autoDif/')
        % set up minFunc's parameters
        params = [log(obj.k_obj.k_params); log(obj.noise)];
        
        % parse options
        if ~isfield(obj.param_struct, 'useMex')
          options.useMex = 0;
        else          
          options.useMex = obj.param_struct.useMex;
        end  
        if ~isfield(obj.param_struct, 'Display')
          obj.param_struct.Display = 'off';
          options.Display = 'off';
        else
          options.Display = obj.param_struct.Display;
        end  
        if ~isfield(obj.param_struct, 'solver')
          obj.param_struct.solver = 'primal';        
        end  
        if ~isfield(obj.param_struct, 'DerivativeCheck')
          obj.param_struct.DerivativeCheck = 'off';        
        else
          options.DerivativeCheck = obj.param_struct.DerivativeCheck;
        end  
        
        if ~strcmp(obj.param_struct.solver, 'primal') && ...
           ~strcmp(obj.param_struct.solver, 'dual')
         disp('Incorrect choice for optimizer.solver: resorting to primal')
         obj.param_struct.solver = 'primal';
        end
                                
        options.MaxIter = 350;      
        opt_params = minFunc(@kernelObserver.negative_log_likelihood_rks, ...
                             params, options, obj.nbases_rks, obj.ndim, obj.seed, obj.sort_mat,...
                             data, obs, obj.k_obj.k_name, 'RandomKitchenSinks', ... 
                             obj.param_struct.solver);
        
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
      weights = kernelObserver.solve_tikhonov(mapped_data, transpose(obs),...
                                              obj.noise^2);  % solve for weights
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
    
    function K = get_deriv(obj, data)
      %  Given new data, map it to feature space induced by kernel, and
      %  compute the gradient of the data. 
      %
      %  Inputs:
      %    data  - dim x nsamp data matrix, passed in columnwise
      %
      %  Outputs:
      %    K   - nsamp x ncent kernel matrix 
      K = obj.mapper.get_deriv(data);
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
        
    function [mapper] = create_map(obj)
      %  Given the current kernel object and basis centers, construct a
      %  FeatureMap object and store it as the internal variable 'mapper'.
      %
      %  Inputs:
      %    -none
      %
      %  Outputs:
      %    -none       
      mapper = kernelObserver.FeatureMap('RandomKitchenSinks');
      map_struct.nbases = obj.nbases_rks; map_struct.ndim = obj.ndim;
      map_struct.seed = obj.seed; map_struct.kernel_obj = obj.k_obj;
      map_struct.sort_mat = obj.sort_mat; 
      mapper.fit(map_struct);
    end      
    
    function params_out = get_params(obj)
      %  Return the kernel and observation noise parameters as one vector.
      %
      %  Inputs:
      %    -none
      %
      %  Outputs:
      %    -params_out - [kernel_parameters, noise]
      params_out = obj.params_final;
    end      
    
    function set_params(obj, params_in)
      %  Return the kernel and observation noise parameters as one vector.
      %
      %  Inputs:
      %    -params_in - [kernel_parameters, noise]
      %    
      %  Outputs:
      %    -none
      obj.params_final = params_in; 
      obj.k_obj.k_params = params_in(1);
      obj.noise = params_in(2);
      obj.mapper = obj.create_map(); % create feature map using centers and kernel
    end          
    
    function mval = get(obj,mfield)
      % Get a requested member variable.
      %      
      switch(mfield)
        case {'noise'}
          mval = obj.noise;
        case {'weights'}
          mval = obj.weights;
        case {'sigma'}
          mval = obj.sigma;          
        case {'nbases'}
          mval = obj.nbases;
        case {'nbases_rks'}
          mval = obj.nbases_rks;
        case {'params_final'}
          mval = obj.params_final;                    
        case {'mapper'}
          mval = obj.mapper;
        case {'nparams'}
          mval = obj.nparams;
        otherwise                    
           disp('wrong variable name')
      end      
    end  
    
    function set(obj, mfield, mval)
      %
      %  Set a requested member variable.
      %      
      switch(mfield)
        case {'sigma'}
          obj.sigma = mval;
        case {'noise'}
          obj.noise = mval;
        case {'weights'}
          obj.weights = mval;          
        otherwise
           disp('wrong variable name')
      end    
    
  end
        
  end

end