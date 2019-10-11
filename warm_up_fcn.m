function [Hk,Lambda,Nk,D,best_H]=warm_up_fcn(d,warm_up,left_bound,right_bound,fcn_name,noise_std)
%--------------------------------------------------------------------------
% 'warm_up_fcn'
% performs the warm up period by choosing the Sobol sequence as the initial
% sample points
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% Hk          : performance estimations
% Lambda      : sampled solutions
% Nk          : number of times shrinking balls being hit
% D           : distance matrix of sampled solutions
% best_H      : best true objective values found
%
% Input arguments
% ---------------
% d           : dimension of the search region
% warm_up     : number of function evaluations used for warm up 
% left_bound  : left bound of the search region
% right_bound : right bound of the search region
% fcn_name    : test function
% noise_std   : standard deviation of the observation noise
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

% sobol initial points
sobol_all=sobolset(d);
Lambda=net(sobol_all,warm_up); % all sampled solutions
Lambda=left_bound+(right_bound-left_bound)*Lambda; Lambda=Lambda';

Nk(1:warm_up)=ones(1,warm_up);

% D: distance matrix for calculating the surrogate model's weights
D=zeros(warm_up,warm_up);
for i=1:warm_up-1
    for j=i+1:warm_up
        D(i,j)=norm(Lambda(:,i)-Lambda(:,j))^3;
    end
end
D=D+D';

% We do NOT need to implement the shrinking ball strategy here
% since the observations in the warm up period are far enough
H(1:warm_up)=feval(fcn_name,Lambda); % true objective function values
h(1:warm_up)=H+abs(H).*normrnd(0,noise_std,1,warm_up); % noisy observations
Hk(1:warm_up)=h(1:warm_up); % initial function estimations
best_H(1:warm_up)=max(H(1:warm_up)); % record