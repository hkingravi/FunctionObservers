%============================== solve_chol ================================
%
%  Solve linear system AX = B using Cholesky factorization of A already 
%  available to user. Identical to GPML toolbox. 
% 
%  Inputs:
%    L  - Cholesky factorization of A, i.e. A = LL'
%    B  - output for linear system to solve, i.e. X = A\B
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

