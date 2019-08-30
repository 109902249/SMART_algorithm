function value=griewank(X)
%------------------------------------------------------------
% 'griewank' is the scaled and shifted Griewank function
% for Nonlinear Optimization.
% http://infinity77.net/global_optimization/test_functions_nd_G.html#go_benchmark.Griewank
%
% domain: -10<=x_i<=10 for i=1,...,d                  
%                  
% f_max = -1
% x_max = ones(d,1)   
%-----------------------------------------------------------
X=X-1;
D=size(X);
A=sum(X.^2);
B=zeros(D(1),1);
for i=1:D(1)
    B(i)=i;
end
C=cos(X./sqrt(B*ones(1,D(2))));
F=prod(C,1);
value=1/400*A-F+ones(1,D(2));
value = -value*10-1;
end
