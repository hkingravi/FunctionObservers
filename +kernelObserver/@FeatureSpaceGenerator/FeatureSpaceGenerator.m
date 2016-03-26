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
    %kernel_model = [];
    centers_est = [];
    k_type = [];
    bandwidth_init = [];
    noise_init = [];
    optimizer = []; 
  end
    
  % hidden variables 
  properties (Access = protected)    
    
  end
  
  % class methods 
  methods    
    function obj = FeatureSpaceGenerator(centers_est, k_type, bandwidth_init, ...
                                         noise_init, optimizer)
      %  Constructor for FeatureSpaceGenerator.
      %
      %  Inputs:
      %    kernel_model  - object for kernel model, 
      %                    conforming to API
      %
      %  Outputs:
      %    -none 
      %obj.kernel_model = kernel_model;
      obj.centers_est = centers_est;
      obj.k_type = k_type;
      obj.bandwidth_init = bandwidth_init;
      obj.noise_init = noise_init;
      obj.optimizer = optimizer;
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
          'Exiting due to irrecoverable error.');
        throw(err);        
      end
      %params = zeros(obj.kernel_model.get('nparams'), nsteps);
      params = zeros(2, nsteps);
           
      % iterate through data and apply learner
      for i=1:nsteps  
        disp(['Current step in time series: ' num2str(i)])
        kernel_model = kernelObserver.RBFNetwork(obj.centers_est, obj.k_type, ...
                                                 obj.bandwidth_init, ...
                                                 obj.noise_init, obj.optimizer);
        kernel_model.debug_mode = 'on';
        kernel_model.fit(data(:, :, i), obs(:, :, i));
        params(:, i) = kernel_model.get_params();
      end      
    end
    % end methods
  end 
end