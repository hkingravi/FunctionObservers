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
    fspace_model     = [];
    param_stream     = [];
  end
  
  % class methods 
  methods    
    function obj = FeatureSpaceGenerator(fspace_model)
      %  Constructor for FeatureSpaceGenerator.
      %
      %  Inputs:
      %    fspace_model     - feature space model: examples include the
      %                       RBFNetwork and RandomKitchenSinks classes.
      %
      %  Outputs:
      %    -none       
      obj.fspace_model = fspace_model;
    end  
                
    function [params] = fit(obj, data_cell, obs_cell)
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
        nsteps = size(data_cell, 2);
        nsteps2 = size(obs_cell, 2);
        assert(nsteps == nsteps2);
      catch ME
        disp([ME.message '!'])
        err = MException('VerifyInput:OutOfBounds', ...
          'Number of data locations and observations must match: ',...
          'see documentation.');
        throw(err);        
      end      
      params = zeros(obj.fspace_model.get('nparams'), nsteps);
           
      % iterate through data and apply learner
      for i=1:nsteps  
        disp(['Current step in time series: ' num2str(i)])
        if i > 1
        end                                       
        obj.fspace_model.debug_mode = obj.display;
        obj.fspace_model.fit(data_cell{i}, obs_cell{i});
        params(:, i) = obj.fspace_model.get_params();
      end      
    end
           
    function mval = get(obj, mfield)
      % Get a requested member variable.
      switch(mfield)
        case {'fspace_model'}
          mval = obj.fspace_model; 
        otherwise
          disp('wrong variable name')
      end
    end
    
    function set(obj, mfield, mval)
      %
      %  Set a requested member variable.
      %
      switch(mfield)
        case {'fspace_model'}
          obj.fspace_model = mval;
        otherwise
          disp('wrong variable name')
      end
    end

    % end methods
  end 
end