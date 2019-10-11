function [Hk,Nk]=performance_estimation(Hk,Nk,cur_h,Lambda,k,rk,gamma1)
%--------------------------------------------------------------------------
% 'performance_estimation' 
% updates the estimation of the true objective function values 
% based on the current noisy observation cur_h via an 
% asynchronous shrinking ball method
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% Hk       : current performance estimations
% Nk       : current number of times shrinking balls being hit
%
% Input arguments
% ---------------
% Hk       : previous performance estimations
% Nk       : previous number of times shrinking balls being hit
% cur_h    : current noisy observation
% Lambda   : all sampled solutions
% k        : iteration counter
% rk       : shrinking ball radius
% gamma1   : constant power in beta=1./Nk.^gamma1
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

% indices that Lambda(:,I_xk) hits the ball B(xk,rk)
I_xk=find(vecnorm(Lambda(:,1:k-1)-Lambda(:,k))<rk);
% initial estimation at x_k
Hk(k)=mean(cat(2,Hk(I_xk),cur_h));
% update performance estimations if x_i hits the current ball
Nk(I_xk)=Nk(I_xk)+1;
beta=1./Nk.^gamma1;
Hk(I_xk)=(1-beta(I_xk)).*Hk(I_xk)+beta(I_xk)*cur_h;

end