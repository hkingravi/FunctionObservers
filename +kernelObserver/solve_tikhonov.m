%============================= solve_tikhonov =============================
%
%  Given features and a regularization parameter, solve Tikhonov problem
%   AX + lambdaI = Y
% 
%  Inputs:
%    A       - N x M matrix
%    Y       - N x D matrix
%    lambda  - positive scalar
%
%  Outputs:
%    X       - M x D matrix
%
%============================= solve_tikhonov =============================
%
%  Name:      solve_tikhonov.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2016/03/31
%  Modified:  2016/03/31
%
%============================= solve_tikhonov =============================
function X = solve_tikhonov(A, Y, lambda)
if lambda < 0 || size(lambda, 1)*size(lambda, 2) ~= 1
  exception = MException('VerifyInput:OutOfBounds', ...
    ' lambda must be positive scalar');
  throw(exception);
end
jitter = 1e-7;

ncent = size(A, 2);
Pmat = transpose(A)*A + (lambda + jitter)*eye(ncent);
L = chol(Pmat);
Ymat = transpose(A)*Y;
X = kernelObserver.solve_chol(L, Ymat);

end

