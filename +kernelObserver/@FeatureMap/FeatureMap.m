%============================== FeatureMap ================================
%
%  Generate feature map object. Implements API similar to scikit-learn's 
%  transformers. 
% 
%  Inputs:
%    fmap_type  - string indication feature map: 
%                 {"RBFNetwork", "RandomKitchenSinks"}.
%    map_struct - struct with mapping parameters
%
%  Outputs:
%               -see functions
%
%============================== FeatureMap ================================
%
%  Name:      FeatureMap.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2016/03/19
%  Modified:  2016/03/19
%
%============================== FeatureMap ================================
classdef FeatureMap < handle 
  % class properties   
  properties (Access = public)        
    
  end
    
  % hidden variables 
  properties (Access = protected)    
    model_type = [];
    basis = [];
    mapper = [];
  end
  
  % class methods 
  methods    
    function obj = FeatureMap(model_type)
      %  Constructor for FeatureMap.
      %
      %  Inputs:
      %    model_type  - string indication feature map: 
      %                 {"RBFNetwork", "RandomKitchenSinks"}.
      %
      %  Outputs:
      %    -none
      if ~strcmp(model_type, 'RBFNetwork') &&...
         ~strcmp(model_type, 'RandomKitchenSinks')
        exception = MException('VerifyInput:OutOfBounds', ...
                               ' incorrect choice of optimizer');
        throw(exception);
      end
      
      obj.model_type = model_type; 
    end  
                
    function fit(obj, map_struct)
      %  Given a struct with mapping parameters, construct feature map. 
      %
      %  Inputs:
      %    data  - dim_in x nsamp x t time-series data matrix
      %    map_struct - struct with mapping parameters
      %
      %  Outputs:
      %    -none
      if strcmp(obj.model_type, 'RBFNetwork')
        if ~isfield(map_struct, 'centers') || ...
           ~isfield(map_struct, 'kernel_obj')
          exception = MException('VerifyInput:OutOfBounds', ...
                      ' map_struct missing fields centers or kernel_obj');
          throw(exception);
        else
          obj.basis = map_struct.centers;
          obj.mapper = map_struct.kernel_obj; 
        end
      elseif strcmp(obj.model_type, 'RandomKitchenSinks')
      end
       
    end
    
    function [mapped_data] = transform(obj, data)
      %  Map data from input domain to feature space. 
      %
      %  Inputs:
      %    data - dim_in x nsamp data matrix
      %
      %  Outputs:
      %    mapped_data - dim_out x nsamp mapped data matrix
      if strcmp(obj.model_type, 'RBFNetwork')
        mapped_data = kernelObserver.generic_kernel(obj.basis,...
                                                    data, obj.mapper);
      elseif strcmp(obj.model_type, 'RandomKitchenSinks')
      end       
    end    

    function [map_deriv] = get_deriv(obj, data)
      %  Get derivative of feature map with respect to 
      %  parameters of the feature map. This will return a 
      %  multidimensional array of parameter derivative 
      %  matrices, with a list of strings indicating which 
      %  index corresponds to which derivative. 
      %
      %  Inputs:
      %    data - dim_in x nsamp data matrix
      %
      %  Outputs:
      %    param_mats - dim_out x nsamp x nparam parameter derivatives matrix
      %    param_names - 1 x nparam parameter derivatives names
      if strcmp(obj.model_type, 'RBFNetwork')
        % compute matrices associated to derivative 
        mapped_data = kernelObserver.generic_kernel(obj.basis,...
                                                    data, obj.mapper);
        map_deriv = mapped_data.*kernelObserver.dist_mat(obj.basis,...
                                                         data);             
      elseif strcmp(obj.model_type, 'RandomKitchenSinks')
        % stub
      end       
    end        
    
    function mval = get(obj,mfield)
      % Get a requested member variable.
      %
      switch(mfield)
        case {'centers'}
          mval = obj.centers;
        case {'basis'}
          mval = obj.basis;
        case {'mapper'}
          mval = obj.mapper;
        otherwise
          disp('wrong variable name')
      end
    end
    % end methods
  end 
end