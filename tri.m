function value=tri(X)


A=(X-.9).^2;
B=7*A;
C=14*A;
F=8*sin(B).^2;
G=6*sin(C).^2;
value=sum(F+G+A);


value=-value/10-1;
end