function value=qing_test_fcn(X)
%------------------------------------------------------------
% 'qing_test_fcn' is the scaled and shifted Qing function
% for Nonlinear Optimization.
% http://infinity77.net/global_optimization/test_functions_nd_Q.html#go_benchmark.Qing
%
% domain: -10<=x_i<=10 for i=1,...,d                  
%                  
% f_max = -1
% x_max \in {xi=+-\sqrt{i},i=1,...,d}  
%-----------------------------------------------------------
s = size(X);
I = repmat((1:s(1))',1,s(2));
value = sum( (X.^2- I).^2 );
value = -value*1e-2 - 1;
end