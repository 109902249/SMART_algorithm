function [mu_new,var_new]=...
    truncated_inv_mean_para_fcn(left_bound,right_bound,mu_old,var_old,eta_x_new,eta_x2_new)
% 'truncated_inv_mean_para_fcn' calculates the sampling parameter based on
% the mean parameter function value. It is the inverse function of the mean
% parameter function. Since we do not have an analytical form of the
% solution, we use an iterative method to approximate the solution with a
% preset precision.
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% mu_new  : the corresponding sampling mean
% var_new : the corresponding sampling variance
%
% Input arguments:
% ---------------
% left_bound  : left bound of the search region
% right_bound : right bound of the search region
% mu_old      : previous sampling mean
% var_old     : previous sampling variance
% eta_x_new   : current mean parameter function value on X
% eta_x2_new  : current mean parameter function value on X^2
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------
precision=1e-3;
iter=0;
while true
    iter=iter+1;
    
    pdf_l=normpdf(left_bound,mu_old,var_old.^0.5); 
    pdf_u=normpdf(right_bound,mu_old,var_old.^0.5);

    cdf_l=normcdf(left_bound,mu_old,var_old.^0.5); 
    cdf_u=normcdf(right_bound,mu_old,var_old.^0.5);

    A=(pdf_l-pdf_u)./(cdf_u-cdf_l);
    B=1+((left_bound-mu_old)./(var_old.^0.5).*pdf_l-...
        (right_bound-mu_old)./(var_old.^0.5).*pdf_u)./(cdf_u-cdf_l);
 
    var_new=(eta_x2_new-eta_x_new.^2)./(B-A.^2);
    mu_new=eta_x_new-A.*var_old.^0.5;
    
    if norm(mu_new-mu_old)<precision && norm(var_new-var_old)<precision
        break
    end
    
    mu_old=mu_new;
    var_old=var_new;
end
