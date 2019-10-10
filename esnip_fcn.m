function exp_surrogate_num_int_pts=...
    esnip_fcn(x_num_int,numNumInt,weight,Lambda,t,mu,var,left_bound,right_bound)
% 'esnip_fcn'
% calculates the exponential value of the surrogate model at numerical integration points
%--------------------------------------------------------------------------
% Output argument
% ----------------
% exp_surrogate_num_int_pts
%
% Input arguments
% ---------------
%
% x_num_int   : points used for numerical intergation
% numNumInt   : number of points for numerical integration
% weight      : coefficents of the surrogate model
% Lambda_ik   : sampled solutions
% t           : annealing temperature
% mu          : sampling mean
% var         : sampling variance
% left_bound  : left bound of the search region
% right_bound : right bound of the search region
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

exp_surrogate_num_int_pts=[];
% P: change of measure
C=normcdf(right_bound,mu,var.^0.5)-normcdf(left_bound,mu,var.^0.5);
P=prod(normpdf(x_num_int,repmat(mu,1,numNumInt),repmat(var.^0.5,1,numNumInt))./repmat(C,1,numNumInt));
% Parallel computing could be implemented here.
for i=1:numNumInt
    exp_surrogate_num_int_pts(i)=exp(sum(weight'.*vecnorm(x_num_int(:,i)-Lambda).^3)/t)/P(i);
end

% normalization
exp_surrogate_num_int_pts=exp_surrogate_num_int_pts/max(exp_surrogate_num_int_pts);
    
end