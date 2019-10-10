function value=ackley_test_fcn(X)
%--------------------------------------------------------------------------
% 'ackley_test_fcn'
% is the scaled and shifted Ackley function for Nonlinear Optimization
% http://infinity77.net/global_optimization/test_functions_nd_A.html#go_benchmark.Ackley
%
% domain: -10<=x_i<=10 for i=1,...,d                  
%                  
% f_max = -1
% x_max = ones(d,1)   
%--------------------------------------------------------------------------
X=X-1;
a=20;
b=.5;
D=size(X);
n=D(1);
A1=X.^2;
A=a*exp(-b*sqrt(sum(A1)/n));
B1=cos(2*pi*X);
B=exp(sum(B1)/n);
value=-a-exp(1)+A+B;
value=value-1;
end