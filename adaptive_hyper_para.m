function [alpha,rk,t,numNumInt]=adaptive_hyper_para(k,warm_up,ca,gamma0,cr,gamma2,best_H)
%--------------------------------------------------------------------------
% 'adap_hyper_para'
% calculates the adaptive hyperparameters required in the current iteration k
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% alpha       : learning rate for updating the mean parameter function
% rk          : shrinking ball radius
% t           : annealing temperature
% numNumInt   : number of points for numerical integration
%
% Input arguments
% ---------------
% k           : iteration counter
% warm_up     : number of function evaluations used for warm up
% ca          : constant shift in alpha=1/(k-warm_up+ca)^gamma0 
% gamma0      : constant power in alpha=1/(k-warm_up+ca)^gamma0
% cr          : constant shift in rk=cr/log(k-warm_up+1)^gamma2
% gamma2      : constant power in rk=cr/log(k-warm_up+1)^gamma2
% best_H      : current best objective function value found
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

% alpha: learning rate for updating the mean parameter function
alpha=1/(k-warm_up+ca)^gamma0;
% rk: shrinking ball radius
rk=cr/log(k-warm_up+1)^gamma2;
% t: annealing temperature
t=.1*abs(best_H)/log(k-warm_up + 1);
% numNumInt: number of points for numerical integration
numNumInt=max(100,floor((k-warm_up)^.5));

end