%============================= kernelObj ==================================
%
%  Data encapsulator for kernels. Encodes kernel name and parameters. 
%
%  Reference(s):   
%    -Rahimi & Recht - Random Features for Large-Scale Kernel Machines
% 
%  Inputs:
%    sigma      - parameter for kernel
%    ncent      - number of bases: you will generate a map TWICE this
%                 length!
%    dim        - dimensionality of data
%
%  Outputs:
%               -see functions
%
%============================= kernelObj ==================================
%
%  Name:      kernelObj.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2015/10/13
%  Modified:  2015/10/13
%
%============================= kernelObj ==================================
classdef kernelObj < handle 
  % class properties   
  properties (Access = public)        
    k_name = [];
    k_params = [];
  end
    
  % class methods 
  methods
    
    function obj = kernelObj(k_name, k_params)
      %  Constructor for kernelObj.
      %
      %  Inputs:
      %    k_name      - 
      %    k_params    - 
      %
      %  Outputs:
      %    -none  
      if ~ischar(k_name)
        exception = MException('VerifyInput:OutOfBounds', ...
                               'Kernel type must a string');
        throw(exception);
      elseif ~strcmp(k_name, 'gaussian') && ~strcmp(k_name, 'laplacian')...
                                         && ~strcmp(k_name, 'polynomial')...
                                         && ~strcmp(k_name, 'sqexp')
        exception = MException('VerifyInput:IncorrectInput', ...
          'Kernel name must be gaussian, laplacian, sqexp, or polynomial');
        throw(exception);
      end
      if ~isnumeric(k_params)
        exception = MException('VerifyInput:IncorrectType', ...
                               'Kernel parameters must be in matrix form');
        throw(exception);
      end      
      if size(k_params, 1) ~= 1
        exception = MException('VerifyInput:IncorrectType', ...
                               'Kernel parameters must be 1 x p matrix');
        throw(exception);
      end  
      if size(k_params, 2) > 1
         disp(['kernelObj:WARNING: multidimensional bandwidth passed in:' ...
               ' using first element of array'])
         k_params = k_params(1, 1);
      end  
      
      obj.k_name = k_name;            
      obj.k_params = k_params;      
    end   

  end
  
end