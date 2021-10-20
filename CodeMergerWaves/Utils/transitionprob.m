function [P,ip] = transitionprob(T,M,s,rm)
% Draws transition probs 

%% INPUT

% T size obs
% M numbers of regimes
% s state variable (1,...,M regimes). Dimension (TKxM)
% rm Dirichlet

%% OUTPUT

% P transition matrix
% ip invariant prob
% transprob empirical transition prob

P=zeros(M,M);
nt=[s(1:T-1) s(2:T)];
transprob=zeros(M,M);
for im=1:M
    for j=1:M
        transprob(im,j)=size(nt(nt(:,1)==im & nt(:,2)==j,:),1);              
    end
    P(im,:)=dirichletrnd(1,M,rm(im,:)+transprob(im,:),0);   
end
ip=[(P'-eye(M)); ones(1,M)];
if det(ip'*ip)~=0
    ip=inv(ip'*ip)*ip';
    ip=ip(:,M+1);
else
    ip=ones(M,1)/M;
end
