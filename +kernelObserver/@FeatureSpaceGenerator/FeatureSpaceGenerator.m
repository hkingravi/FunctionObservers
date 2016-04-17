%======================== FeatureSpaceGenerator ===========================
%
%  Given an object instantiation of a finite-dimensional kernel model, and
%  a time-series with inputs and outputs, this class will infer the
%  parameters for the kernel model, train weights for each step in the time
%  series, and return the kernel model and the weights as a time-series. 
%
%  To make training simple, each kernel model must conform to an API. The
%  curent version of the code only deals with single-dimensional outputs. 
% 
%  Inputs:
%    kernel_model  - object for finite-dimensional kernel model. 
%
%  Outputs:
%               -see functions
%
%======================== FeatureSpaceGenerator ===========================
%
%  Name:      FeatureSpaceGenerator.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2016/03/13
%  Modified:  2016/03/13
%
%======================== FeatureSpaceGenerator ===========================
classdef FeatureSpaceGenerator < handle 
  % class properties   
  properties (Access = public)   
    display          = 'on';  % whether to display output
  end
    
  % hidden variables 
  properties (Access = protected)    
    fspace_type      = [];
    fspace_model     = [];
    basis            = []; 
    mapper_type      = []; 
    params_init      = []; 
    params_optimizer = [];     
  end
  
  % class methods 
  methods    
    function obj = FeatureSpaceGenerator(fspace_type, basis, mapper_type,...
                                         params_init, optimizer)
      %  Constructor for FeatureSpaceGenerator.
      %
      %  Inputs:
      %    fspace_type      - string indicating generator for feature space:
      %                       {'RBFNetwork', 'RandomKitchenSinks'}
      %    basis            - dim x ncent basis for feature space class
      %    mapper_type      - string indicating kernel or function generating
      %                       map in FeatureMap (adding support for trees in 
      %                       the future)
      %    params_init      - initial parameters for mapper object: will be
      %                       used as initialization for params_optimizer: 
      %                       If optimizer is 
      %                       'cv': pass in nparams x param_vals array
      %                       'likelihood': pass in nparams x 1 vector
      %    params_optimizer - 'cv': cross-validation
      %                       'likelihood': optimizer using marginal
      %                                     likelihood
      %
      %  Outputs:
      %    -none       
      obj.fspace_type = fspace_type;
      obj.basis = basis;
      obj.mapper_type = mapper_type;
      obj.params_init = params_init; 
      obj.params_optimizer = optimizer;
    end  
                
    function [params] = fit(obj, data, obs)
      %  Given time-series data, infer parameters of kernel model as
      %  time-series. Infer final parameters 
      %
      %  Inputs:
      %    data  - dim_in x nsamp x t time-series data matrix
      %    obs   - dim_out x nsamp x t time-series observation matrix
      %
      %  Outputs:
      %    -none 
      try
        [~, nsamp, nsteps] = size(data);
        [~, nsamp2, nsteps2] = size(obs);
        assert(nsteps == nsteps2 && nsamp == nsamp2);
      catch ME
        disp([ME.message '!'])
        err = MException('VerifyInput:OutOfBounds', ...
          'Number of data locations and observations must match: ',...
          'see documentation.');
        throw(err);        
      end

      
      params = zeros(kernel_model.nparams, nsteps);
           
      % iterate through data and apply learner
      for i=1:nsteps  
        disp(['Current step in time series: ' num2str(i)])
        if i > 1
        end                                       
        kernel_model.debug_mode = obj.display;
        kernel_model.fit(data(:, :, i), obs(:, :, i));
        params(:, i) = kernel_model.get_params();
      end      
    end
    
    function [mapper] = create_map(obj)
      %  Given current parameters, return model representing feature space.
      %
      %  Inputs:
      %    -none
      %
      %  Outputs:
      %    -none       
      % initialize model
      if strcmp(obj.fmap_type, 'RBFNetwork')
        kernel_model = kernelObserver.RBFNetwork(obj.basis, obj.mapper_type, ...
                                                 obj.params_init(1), ...
                                                 obj.params_init(2), ...
                                                 obj.optimizer);                                               
      elseif strcmp(obj.fmap_type, 'RandomKitchenSinks')        
        %------------------- STUB ---------------
      else
        err = MException('VerifyInput:OutOfBounds', ...
          'Invalid choice for fmap_type : see documentation.');
        throw(err);        
      end  
    end  
    
    
    function mval = get(obj,mfield)
      % Get a requested member variable.
      switch(mfield)
        case {'fmap_type'}
          mval = obj.fmap_type;
        case {'basis'}
          mval = obj.basis;
        case {'mapper_type'}
          mval = obj.mapper_type;
        case {'params_init'}
          mval = obj.params_init;
        case {'params_optimizer'}
          mval = obj.params_optimizer;
        otherwise
          disp('wrong variable name')
      end
    end
    
    function set(obj,mfield,mval)
      %
      %  Set a requested member variable.
      %
      switch(mfield)
        case {'fmap_type'}
          obj.fmap_type = mval;
        case {'basis'}
          obj.basis = mval;
        case {'mapper_type'}
          obj.mapper_type = mval;
        case {'params_init'}
          obj.params_init = mval;          
        case {'params_optimizer'}
          obj.params_optimizer = mval;                    
        otherwise
          disp('wrong variable name')
      end
    end

    % end methods
  end 
end