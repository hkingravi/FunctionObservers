%============================== dist_mat ==================================
%  
%  This code takes as input two data matrices, and returns a distance
%  matrix. 
%
%  Reference(s): 
% 
%  INPUT:
%    data1	    - d x n data matrix, with each column as an observation. 
%    data2	    - d x m data matrix, with each column as an observation.
%
%  OUTPUT:
%               - n x m distance matrix 
%
%============================== dist_mat ==================================
%
%  Name:        dist_mat.m
%
%  Author:      Hassan A. Kingravi
%
%  Created:  	2016/03/20
%  Modified: 	2016/03/20
%
%============================== dist_mat ==================================
function val =  dist_mat(x, y)

d = x'*y;
dx = sum(x.^2,1);
dy = sum(y.^2,1);
val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;

end
