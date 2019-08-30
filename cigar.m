function value=cigar(X)
%------------------------------------------------------------
% 'cigar' is the scaled and shifted Bent Cigar function
% for Nonlinear Optimization.
% http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.Cigar
%
% domain: -10<=x_i<=10 for i=1,...,d                  
%                  
% f_max = -1
% x_max = ones(d,1)   
%-----------------------------------------------------------
X=X-1;
value=X(1,:).^2+1e6*sum(X(2:end,:).^2);
value=-value*1e-7-1;
end