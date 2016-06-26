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
    seed       = [];    
    model_type = [];
    basis      = [];
    mapper     = [];
    nbases     = [];
    sort_mat   = [];
    ndim       = [];
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
          if ~isfield(map_struct, 'seed')
            obj.seed = 0;
          else
            obj.seed = map_struct.seed;
          end  
          s = RandStream('mt19937ar','Seed', obj.seed);
          RandStream.setGlobalStream(s);
          
          obj.basis = map_struct.centers;
          obj.mapper = map_struct.kernel_obj; 
          obj.nbases = size(map_struct.centers, 2);
          
          if isfield(map_struct, 'sort_mat')
            obj.sort_mat = map_struct.sort_mat; 
          end  
        end
      elseif strcmp(obj.model_type, 'RandomKitchenSinks')
        if ~isfield(map_struct, 'nbases') || ...
           ~isfield(map_struct, 'ndim') || ...
           ~isfield(map_struct, 'kernel_obj') || ...
           ~isfield(map_struct, 'seed') || ...
           ~isfield(map_struct, 'sort_mat')
          exception = MException('VerifyInput:OutOfBounds', ...
            ' map_struct missing fields: see documentation');
          throw(exception);
        else
          obj.seed = map_struct.seed;
          obj.sort_mat = map_struct.sort_mat;          
          s = RandStream('mt19937ar','Seed', obj.seed);
          RandStream.setGlobalStream(s);
          
          bandwidth = map_struct.kernel_obj.k_params(1);
          obj.nbases = map_struct.nbases;
          obj.basis = (1/bandwidth)*randn(map_struct.ndim, obj.nbases);
          if map_struct.ndim == 1 && map_struct.sort_mat == 1            
            disp('Sorting random matrix...')
            obj.basis = sort(obj.basis);
          end
          obj.mapper = map_struct.kernel_obj;          
        end
      end
      obj.ndim = size(obj.basis, 1); 
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
        data_trans = transpose(obj.basis)*data;
        mapped_data = [sin(data_trans); cos(data_trans)]/sqrt(obj.nbases);         
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
        [~, map_deriv] = kernelObserver.generic_kernel(obj.basis,...
                                                       data, obj.mapper);
      elseif strcmp(obj.model_type, 'RandomKitchenSinks')
        data_trans = transpose(obj.basis)*data;
        data_comp = (1/obj.mapper.k_params(1)).*[data_trans; data_trans];
        map_deriv = {};
        map_deriv_curr = [-cos(data_trans); 
                          sin(data_trans)]/sqrt(obj.nbases);         
        map_deriv{1} = map_deriv_curr.*data_comp;
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
        case {'model_type'}
          mval = obj.model_type;          
        case {'seed'}
          mval = obj.seed;   
        case {'sort_mat'}
          mval = obj.sort_mat; 
        otherwise
          disp('wrong variable name')
      end
    end
    % end methods
  end 
end