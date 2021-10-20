function [BetaDraw] = DrawBetas(Data,Y,X,K,filts,mbar,Mbar)

% PURPOSE: draws conditional betas

[p]      = size(X,3);
BetaDraw = zeros(K,p);


for k=1:K
    xi     = find(filts==k);
    Ti     = size(xi,2);
    
    temp  = zeros(p);
    temp1 = zeros(p,1);
    
    for t = 1:Ti
    temp  = temp  + squeeze(X(xi(t),1:Data(t)+1,:))'*squeeze(X(xi(t),1:Data(t)+1,:));
    temp2 =  squeeze(X(xi(t),1:Data(t)+1,:))'*squeeze(Y(xi(t),1:Data(t)+1))';
    if Data(t)==0
        temp2 = temp2';
    end    
    temp1 = temp1 + temp2;
    end        
    
    iMbar  = inv(Mbar(p*(k-1)+1:p*k,p*(k-1)+1:p*k)); 
    Mstar  = inv(iMbar + temp);
    mstar  = Mstar*(iMbar*mbar(k,:)' + temp1);
%     
        [~, h]  = chol(Mstar);
        
        if h ~= 0
            [V,D] = eig(Mstar);
            V1 = V(:,1);
            C2 = Mstar + V1*V1'*(eps(D(1,1))-D(1,1));
            Mstar = C2;
        end

    beta = rMNorm(mstar,Mstar,1)';
    BetaDraw(k,:) = beta;
    
end

