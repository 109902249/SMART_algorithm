function value=bohachevsky_test_fcn(X)
%------------------------------------------------------------
% 'bohachevsky_test_fcn' is the scaled and shifted Bohachevsky function
% for Nonlinear Optimization.
% http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bohachevsky
%
% domain: -10<=x_i<=10 for i=1,...,d                  
%                  
% f_max = -1
% x_max = ones(d,1)   
%-----------------------------------------------------------
X=X-1;
value=sum(X(1:end-1,:).^2+2*X(2:end,:).^2 ...
    -0.3*cos(3*pi*X(1:end-1,:))-0.4*cos(4*pi*X(2:end,:))+0.7);
value=-value*1e-1-1;
end