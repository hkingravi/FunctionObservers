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
%  Modified:  2016/05/10
%
%========================== measurement_operator ==========================
function [Kmat, meas_basis,...
          meas_inds] =  measurement_operator(meas_type, nmeas, fmap, data)

if ~strcmp(meas_type, 'random') &&...
    ~strcmp(meas_type, 'rational')
  exception = MException('VerifyInput:OutOfBounds', ...
    ' incorrect choice of measurement type:',...
    ' see documentation');
  throw(exception);
end

% basis = fmap.get('basis');

if strcmp(meas_type, 'random')
  % randomly select subset of data to construct operator  
  nsamp = size(data, 2);
  rand_inds = randperm(nsamp);
  meas_inds = rand_inds(1:nmeas);
  meas_basis = data(:, meas_inds);
  
  % just check if matrix is sorted: only really used for pretty plots
  if fmap.get('sort_mat') == 1
    meas_basis = sort(meas_basis);  
  end 
  
  Kmat = fmap.transform(meas_basis);
elseif strcmp(meas_type, 'rational')
  exception = MException('VerifyInput:OutOfBounds', ...
    ' rational map currently not implemented:',...
    ' see documentation');
  throw(exception);
end

end