%=========================== KernelObserver ===============================
%
%  Given an object instantiation of a finite-dimensional kernel model, and
%  a time-series with inputs and outputs, this class will infer the
%  parameters for the kernel model, train weights for each step in the time
%  series, and compute the measurement-dynamics operator pair (K, A) from
%  the data. We follow the steps below. 
%
%  Step 1: Load mapping object for feature space (e.g. RandomKitchenSinks)
%  Step 2: Run FeatureSpaceGenerator on time-series data, to get optimal
%          feature space parameters (we will, for now, fit a mean to the
%          time series)
%  Step 3: Map the same time series to using this new feature space object,
%          and learn the weights of the regression. 
%  Step 4: Utilize matrix-valued least squares to infer A operator.
%  Step 5: Utilize other methods such a random placement or measurement map
%          to construct K operator.
%  Step 6: Initialize covariance matrix of weights (or use scaled identity),
%          and utilize this matrix to create a kernel Kalman filter. 
%  Step 7: Now that you have the observer, you can use it for prediction
%          etc. 
% 
%  Inputs:
%    kernel_model  - object for finite-dimensional kernel model. 
%
%  Outputs:
%               -see functions
%
%=========================== KernelObserver ===============================
%
%  Name:      KernelObserver.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2016/04/21
%  Modified:  2016/04/21
%
%=========================== KernelObserver ===============================
classdef KernelObserver < handle 
  % class properties   
  properties (Access = public)   
    display           = 'on';  % whether to display output
    ret_param_stream  = 'on';
    ret_weight_stream = 'on';
    meas_op_type      = [];    % {'random', 'rational'}
  end
    
  % hidden variables 
  properties (Access = protected)    
    fspace_model     = [];
    nmeas            = [];
    fspace_generator = [];
    param_stream     = [];
    weight_stream     = [];    
    params           = [];
    reg_parm         = 0.01;
    A                = [];  % dynamics operator
    K                = [];  % observation operator    
  end
  
  % class methods 
  methods    
    function obj = KernelObserver(fspace_model, nmeas, meas_op_type)
      %  Constructor for KernelObserver.
      %
      %  Inputs:
      %    fspace_model     - feature space model: examples include the
      %                       RBFNetwork and RandomKitchenSinks classes
      %    nmeas            - number of measurements (sensing locations)      
      %    meas_op_type     - measurement operator type: 
      %                       {'random', 'rational'}
      %
      %  Outputs:
      %    -none       
      obj.fspace_model = fspace_model;
      obj.fspace_generator = kernelObserver.FeatureSpaceGenerator(obj.fspace_model);
      obj.meas_op_type = meas_op_type;
      obj.nmeas = nmeas;
    end  
                
    function fit(obj, data, obs)
      %  Given time-series data, infer parameters of kernel model as
      %  time-series. Infer final parameters 
      %
      %  Inputs:
      %    data  - dim_in x nsamp x t time-series data matrix
      %    obs   - dim_out x nsamp x t time-series observation matrix
      %
      %  Outputs:
      %    -none       
      param_stream_out = obj.fspace_generator.fit(data, obs);
                  
      % infer ideal parameters by computing mean across parameters, set
      % fspace_model to use these, and infer weights 
      obj.params = mean(param_stream_out, 2);  
      obj.fspace_model.set_params(obj.params);     
      nsteps = size(data, 3);
      ncent = obj.fspace_model.get('nbases');
      weights = zeros(ncent, nsteps);
      for i=1:nsteps
        weights(:, i) = obj.fspace_model.fit_current(data(:, :, i), obs(:, :, i));
      end      
            
      % infer dynamics operator using least-squares
      weight_set = [zeros(ncent, 1), weights(:, 1:end-1)];
      sys_mat = weight_set*transpose(weight_set) + obj.reg_parm*eye(ncent);
      obj.A = transpose(sys_mat\(weight_set*transpose(weights)));      
  
      % retain data if asked
      if strcmp(obj.ret_param_stream, 'on')
        obj.param_stream = param_stream_out;        
      end      
      if strcmp(obj.ret_weight_stream, 'on')
        obj.weight_stream = weights; 
      end
      
      % construct Kalman filter using dynamics operator and measurement
      % operator
      obj.K = kernelObserver.measurement_operator(obj.meas_op_type, obj.nmeas, ...
                                                  obj.fspace_model.get('mapper'));
      
    end    

    function [param_stream] = get_param_stream(obj)
      %  Given time-series data, infer parameters of kernel model as
      %  time-series. Infer final parameters 
      %
      %  Inputs:
      %    data  - dim_in x nsamp x t time-series data matrix
      %    obs   - dim_out x nsamp x t time-series observation matrix
      %
      %  Outputs:
      %    -none       
      param_stream = obj.param_stream;
    end
        
    function mval = get(obj, mfield)
      % Get a requested member variable.
      switch(mfield)
        case {'fspace_model'}
          mval = obj.fspace_model; 
        case {'nmeas'}
          mval = obj.nmeas;           
        case {'param_stream'}
          mval = obj.param_stream;
        case {'reg_parm'}
          mval = obj.reg_parm; 
        case {'A'}
          mval = obj.A;
        case {'K'}
          mval = obj.K;
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
        case {'reg_parm'}
          obj.reg_parm = mval;          
        otherwise
          disp('wrong variable name')
      end
    end

    % end methods
  end 
end