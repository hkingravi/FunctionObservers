%============================== solve_chol ================================
%
%  Given feature mapped data and a regularization parameter, solve linear
%  system and generate weights. 
% 
%  Inputs:
%    L  - Cholesky factorization of A, i.e. A = LL'
%    B  - linear system to solve, i.e. X = A\B
%
%  Outputs:
%               -see functions
%
%============================== solve_chol ================================
%
%  Name:      solve_chol.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2016/03/30
%  Modified:  2016/03/30
%
%============================== solve_chol ================================
function X = solve_chol(L, B)
X = L\(L'\B);
end

