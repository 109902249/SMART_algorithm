function value=pinter(X)
%------------------------------------------------------------
% 'pinter' is the scaled and shifted Pinter function
% for Nonlinear Optimization.
% http://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.Pinter
%
% domain: -10<=x_i<=10 for i=1,...,d                  
%                  
% f_max = -1
% x_max = ones(d,1)   
%-----------------------------------------------------------
X=X-0.5;
D=size(X);
A1=X.^2;
A2=zeros(D(1),1);
for i=1:D(1)
    A2(i)=i;
end
A=A2*ones(1,D(2)).*A1;
B1=X(D(1),:);
B2=X(1:(D(1)-1),:);
B3=X(1,:);
B4=X(2:D(1),:);
B5=[B2;B1];
B6=[B4;B3];
B7=B5.*sin(X)-X+sin(B6);
B=20*(A2*ones(1,D(2))).*sin(B7).^2;
C1=B5.^2-2*X+3*B6-cos(X)+1;
C2=1+(A2*ones(1,D(2))).*C1.^2;
C=(A2*ones(1,D(2))).*log10(C2);
value=sum(A+B+C);
value=-value*1e-2-1;
end