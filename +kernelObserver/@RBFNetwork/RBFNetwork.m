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
    k_obj      = [];        % kernel object, empty at initialization
    noise      = 0.1;       % current noise parameter
    ncent      = []; 
    warnings   = 'on';
    debug_mode = 'off';     
  end
    
  % hidden variables 
  properties (Access = protected)    
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
      %                 must have fields - method -{'none', 'cv'}
      %                                  - nfolds - number of folds for cv
      %
      %  Outputs:
      %    -none
      try 
      % nesting structure to set new values 
      if nargin >=3
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
                ~strcmp(optimizer.method, 'cv')
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

      catch ME
        disp([ME.message])  
        err = MException('VerifyInput:OutOfBounds', ...
                         'Exiting due to incorrect inputs.');
        throw(err);
      end      
    end  
          
    function fit(obj,data, obs)
      %  Given new data (possibly in batch form), update the weight vector
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
      obj.mapper = obj.create_map(); 
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
      try        
        K = obj.mapper.transform(data); 
                
      catch ME
        disp([ME.message '!'])
        err = MException('VerifyInput:OutOfBounds', ...
          'Exiting due to irrecoverable error.');
        throw(err);
      end
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
      try  
        K = obj.mapper.transform(data_test);
        if nargin > 2
          f = transpose(weights_in)*K;
        else
          f = transpose(obj.weights)*K;
        end
                
      catch ME
        disp([ME.message '!'])
        err = MException('VerifyInput:OutOfBounds', ...
          'Exiting due to irrecoverable error.');
        throw(err);
      end
    end    
    
    function [mapper] = create_map(obj)
      mapper = kernelObserver.FeatureMap('RBFNetwork');
      map_struct.centers = obj.centers; map_struct.kernel_obj = obj.k_obj;
      mapper.fit(map_struct);
    end  
    
    function params = get_params(obj)
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