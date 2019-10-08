function value=styblinskiTang_test_fcn(X)
%------------------------------------------------------------
% 'styblinskiTang_test_fcn' is the scaled and shifted Styblinski-Tang function
% for Nonlinear Optimization.
% http://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.StyblinskiTang
%
% domain: -10<=x_i<=10 for i=1,...,d                  
%                  
% f_max = -1
% x_max = -2.9*ones(d,1)   
%-----------------------------------------------------------
value = .5*sum( X.^4 - 16 * X.^2 + 5 * X ) + 39.166*20;
value = -value*1e-2 - 1;
end