% Purpose: Sampling from a mixture of 10 normal distribution to
% approximate a log chi2
%--------------------------------------------------------------------------
% Usage: [res k]= mixtures(lnsig,Y,ind)
%--------------------------------------------------------------------------
% Input: 
%--------------------------------------------------------------------------
% lnsig = [T x 1] vector of log volatilities
% Y     = [T x 1] vector of observations
% ind   = flag to identify the sampling options, 'O' Omori et al. 2007
%--------------------------------------------------------------------------
% Output:
% res   = [T x 1] vector of selected means and stds of the mixture
% k     = [T x 1] vector of drawing from the multinomial
%--------------------------------------------------------------------------
% References:
% Omori, Chib, Shephard, Nakajima (2007), Journal of Econometrics
% " Stochastic volatility with leverage: Fast and efficient likelihood
% inference"
%--------------------------------------------------------------------------

function [mus, ss2s, k]= mixtures(dep,indep)
 
  [N] = size(dep,1);
  
   mu        = [5.09 3.29 1.82 1.24 0.764 0.391 0.0431 -0.306 -0.673 -1.06];
   ss2       = [4.50 2.02 1.10 0.422 0.198 0.107 0.0778 0.0766 0.0947 0.146];  
   q         = [0.00397 0.00396 0.168 0.147 0.125 0.101 0.104 0.116 0.107 0.088]; 
         
   sig       = sqrt(ss2);
  
  temp      = dep - indep;
  w         = zeros(N,size(q,2));
  k         = zeros(N,1);
  mus       = zeros(N,1);
  ss2s      = zeros(N,1);
  
  for i=1:N
  w(i,:)      = normpdf(temp(i),mu,sig).*q;
  if ~(sum(w(i,:)) > 0) || ~all(w(i,:)>=0) % catches missing values
  w(i,:)      = ones(1,size(q,2))*1/size(q,2);
  end
  k(i)        = randsample(size(q,2),1,true,w(i,:));
  mus(i)      = mu(k(i));
  ss2s(i)     = ss2(k(i)); 
  end
   
end  
  
  