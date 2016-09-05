%============================= MLModel ====================================
%
%  Abstract class implementing API that all machine learning models
%  exploiting feature maps must adhere to. Machine learning models in the
%  function observer paradigm consist of a feature map, an optimizer (with
%  an implicit loss function), and a final parameter vector. 
% 
%  Inputs:
%    - none           
%
%  Outputs:
%    - none
%
%============================= MLModel ====================================
%
%  Name:      MLModel.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2016/08/22
%  Modified:  2016/08/22
%
%============================= MLModel ====================================
classdef (Abstract) MLModel < handle
  % class properties   
  properties (Abstract)   
    fmap           % FeatureMap object
    optimizer      % a struct which must contain 'method' field
    params_final
  end
    
  % class methods 
  methods (Abstract) 
    
  end 
end