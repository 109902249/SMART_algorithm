function [eta_x,eta_x2]=truncated_mean_para_fcn(left_bound,right_bound,mu,var)
%--------------------------------------------------------------------------
% 'truncated_mean_para_fcn' calculates the mean parameter function value on
% a truncated domain. The mean parameter function is given by
% m(theta) = E_{sampling distribution}[(X,X^2)]
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% eta_x  : the expectation of X under the sampling distribution
% eta_x2 : the expectation of X^2 under the sampling distribution
%
% Input arguments:
% ---------------
% left_bound  : left bound of the search region
% right_bound : right bound of the search region
% mu          : mean of the sampling distribution
% var         : variance of the sampling distribution
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------
% Truncated coefficients of an independent multivariate normal density 
coef1=(normpdf(left_bound,mu,var.^0.5)-normpdf(right_bound,mu,var.^0.5));
coef1=coef1./(normcdf(right_bound,mu,var.^0.5)-normcdf(left_bound,mu,var.^0.5));
coef2=((left_bound-mu)./var.^0.5).*normpdf(left_bound,mu,var.^0.5);
coef2=coef2-((right_bound-mu)./var.^0.5).*normpdf(right_bound,mu,var.^0.5);
coef2=coef2./(normcdf(right_bound,mu,var.^0.5)-normcdf(left_bound,mu,var.^0.5));

% Outputs
eta_x=mu+coef1.*var.^0.5;
eta_x2=mu.^2+2*mu.*var.^0.5.*coef1+var.*(1+coef2); 