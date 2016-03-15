function [min_row, min_col] = find_min_inds(A)
%FIND_MIN_INDS Find indices associated to minimum value in an array. 
%   Basically an argmin over a matrix. 

[min_vals, min_inds_rows] = min(A);
[~, min_col] = min(min_vals);
min_row = min_inds_rows(min_col);
end

