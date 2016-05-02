%========================== measurement_operator ==========================
%
%  Generate feature map object. Implements API similar to scikit-learn's 
%  transformers. 
% 
%  Inputs:
%    meas_type    - string indicating measurement map: 
%                   {"random", "rational"}.
%    nmeas        - number of measurements required 
%    fmap         - FeatureMap object, encoding kernel, its parameters, 
%                   and the basis
%
%  Outputs:
%               -see functions
%
%========================== measurement_operator ==========================
%
%  Name:      FeatureMap.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2016/04/30
%  Modified:  2016/04/30
%
%========================== measurement_operator ==========================
function [Kmat, meas_basis] =  measurement_operator(meas_type, nmeas, fmap)

if ~strcmp(meas_type, 'random') &&...
    ~strcmp(meas_type, 'rational')
  exception = MException('VerifyInput:OutOfBounds', ...
    ' incorrect choice of measurement type:',...
    ' see documentation');
  throw(exception);
end

basis = fmap.get('basis');
ncent = size(basis, 2);

if strcmp(meas_type, 'random')
  % randomly select subset of basis to construct operator
  rand_inds = randperm(ncent);
  meas_basis = basis(:, rand_inds(1:nmeas));
  Kmat = fmap.transform(meas_basis);
elseif strcmp(meas_type, 'rational')
  exception = MException('VerifyInput:OutOfBounds', ...
    ' rational map currently not implemented:',...
    ' see documentation');
  throw(exception);
end

end