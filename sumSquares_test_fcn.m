function value=sumSquares_test_fcn(X)
%------------------------------------------------------------
% 'sumSquares_test_fcn' is the scaled and shifted sum squares function
% for Nonlinear Optimization.
% https://www.sfu.ca/~ssurjano/sumsqu.html
%
% domain: -10<=x_i<=10 for i=1,...,d                  
%                  
% f_max = -1
% x_max = ones(d,1)   
%-----------------------------------------------------------
X=X-1;
D=size(X);
A=X.^2;
B=zeros(D(1),1);
for i=1:D(1)
    B(i)=i;
end
C=B*ones(1,D(2)).*A;
value=sum(C);
value = -value/100 - 1;
end