function [weight,D]=surrogate_model(Hk,D,Lambda,k)
%--------------------------------------------------------------------------
% 'surrogate_model' constructs the surrogate model that interpolates on all
% sampled points (including the one sampled in the current step).
% We exploit a cubic model: S_k(x)=\sum_{i=1}^{k} weight(i)*||x-xi||^3
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% weight   : weight coefficients of the surrogate model 
% D        : distance matrix of sampled points
%
% Input arguments:
% ---------------
% Hk       : performance estimations
% D        : distance matrix of sampled points
% Lambda   : all sampled solutions
% k        : iteration counter
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

temp=zeros(k,k);
temp(1:k-1,1:k-1)=D;
temp(end,1:end-1)=vecnorm(Lambda(:,1:k-1)-Lambda(:,k)).^3;
% distance matrix is symmetric
temp(1:end-1,end)=temp(end,1:end-1)';
D=temp;
weight=D\Hk(1:k)';

end