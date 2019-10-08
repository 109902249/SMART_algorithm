function [mu_new,var_new,eta_x_new,eta_x2_new]=sampling_updating(mu_old,var_old,left_bound,right_bound,...
        eta_x_old,eta_x2_old,Lambda,weight,numNumInt,t,alpha)
% 'sampling_updating' updates the sampling parameter based on the recursive
% relation derived in [1] and the surrogate model (weight)
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% mu_new      : mean of current multivariate normal sampling distribution
% var_new     : variance of current multivariate normal sampling distribution
% eta_x_new   : current mean parameter function value E[X]
% eta_x2_new  : current mean parameter function value E[X^2]
%
% Input arguments:
% ---------------
% mu_old      : previous sampling mean
% var_old     : previous sampling variance
% left_bound  : left bound of the search region
% right_bound : right bound of the search region
% eta_x_old   : previous mean parameter function value E[X]
% eta_x2_old  : previous mean parameter function value E[X^2]
% Lambda      : all sampled solutions
% weight      : weight coefficents of the surrogate model
% numNumInt   : number of points for numerical integration
% t           : annealing temperature
% alpha       : learning rate for updating the mean parameter function
%--------------------------------------------------------------------------
% REFERENCES
% [1] Qi Zhang and Jiaqiao Hu (2019): Actor-Critic Like Stochastic Adaptive Search
% for Continuous Simulation Optimization. Submitted to Operations Research,
% under review.
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

% generate numNumInt points from the sampling distribution
x_num_int=normt_rnd(mu_old*ones(1,numNumInt),var_old*ones(1,numNumInt),...
        left_bound,right_bound);

% calculate exp_surrogate_num_int_pts based on these samples
exp_surrogate_num_int_pts=esnip_fcn(x_num_int,numNumInt,weight,Lambda,...
        t,mu_old,var_old,left_bound,right_bound);

% calculate E[X] and E[X^2]
int_exp_surrogate=sum(exp_surrogate_num_int_pts);
int_exp_surrogate_x=sum(x_num_int.*exp_surrogate_num_int_pts,2);
int_exp_surrogate_x2=sum(x_num_int.^2.*exp_surrogate_num_int_pts,2);    
EX = int_exp_surrogate_x/int_exp_surrogate;
EX2 = int_exp_surrogate_x2/int_exp_surrogate;

% update the mean parameter function
eta_x_new=eta_x_old+alpha*(EX-eta_x_old);
eta_x2_new=eta_x2_old+alpha*(EX2-eta_x2_old);

% update the sampling parameter
[mu_new,var_new]=truncated_inv_mean_para_fcn(left_bound,right_bound,...
        mu_old,var_old,eta_x_new,eta_x2_new);
    
end