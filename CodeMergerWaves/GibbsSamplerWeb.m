%% Gibbs sampler for the MS Poisson Regression

% This scripts reports the Gibbs sampler for a simplified version of the
% main model in which there is no industry fixed effects and transition
% probabilities are time-invariant. Notice that the bigger the Y the
% higher the computational cost. Similarly, you may need to have a sizable
% computational power to run a large number of MCMC iterations. 

%--------------------------------------------------------------------------
% Housekeeping
%--------------------------------------------------------------------------

clear all; clc; randn('seed',3121), rand('seed',3121), warning off

%--------------------------------------------------------------------------
% Adding folders 
%--------------------------------------------------------------------------

addpath([pwd '/Input/']);
addpath([pwd '/Utils/']);

%--------------------------------------------------------------------------
% Read and select the data aggregate dataset (example in which I use only
% the economic shock as a regressor)
%--------------------------------------------------------------------------
 
[Deals, txt]          = xlsread('DataWeb.xlsx',1,'C2:D1633');
[EconomicShock, txt]  = xlsread('DataWeb.xlsx',2,'C2:D1645');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gibbs Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mcmciter = 1000;
burnin   = 500;
niter    = mcmciter+burnin;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Markov-switching %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Loop

K  = 2;  %number of regimes

nn = 1; % select the industry (example)

[T] = size(Deals(Deals(:,1)== nn,2),1);

X = [EconomicShock(EconomicShock(:,1)== nn,2)];
X = [X(1:end-1,:)./repmat(std(X),T,1)];
X = [ones(T,1) X];

Y = Deals(Deals(:,1)== nn,2);

[T,p] = size(X);  % p; size of the regressors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Prior Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mbar     = zeros(K,p);                        % prior mean on mu ik
M        = [10 zeros(1,p-1);zeros(p-1,1) 10*eye(p-1)];
Mbar     = blkdiag(M,M,M);                    % prior scale on mu ik
P0       = 9*eye(K)+ones(K,K);                % prior on P i

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize Vectors %%%%%%%%%%%%%%%%%%%%%%%%%%

betas           = zeros(niter,K,p);         % Betas
lambdas         = zeros(niter,T);           % Lambdas
P               = zeros(niter,K,K);         % Transition matrix
ip              = zeros(niter,K);           % Ergodic probs
filts           = zeros(niter,T);           % Filtered states
filtp           = zeros(niter,T,K);         % Filtered probs
smoothp         = zeros(niter,T,K);         % Smooth probs
smooths         = zeros(niter,T);           % Smooth probs
clkl            = zeros(niter,T,K);         % Conditional likelihood
tau             = zeros(T,max(Y)+1);        % Interarrival times
Rm              = zeros(T,max(Y)+1);  
Rs              = zeros(T,max(Y)+1);  
Yt              = zeros(T,max(Y)+1);
Xt              = zeros(T,max(Y)+1,p);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gibbs step 0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [P_temp,ip_temp] = transitionprob(T,K,zeros(T,1),P0);

    P(1,:,:)   = P_temp;                % Initial transition probs
    ip(1,:)    = ip_temp;               % Initial ergodic
    sv         = zeros(T,1);            
    pv         = ip_temp;
    
    for t = 1:T
        sv(t) = draw_multinom(pv,1,K,1);    
        pv    = P_temp(sv(t),:)';
    end
    
    filts(1,:) = sv;                    % Initial state    
    
   
    if size(X,2) == 1
    for k = 1:K
        betas(1,k,:)   = mvnrnd(mbar(k,:),eye(p)*1e2);                          % Initial beta
        lambdas(1,:)   = exp(X);                                               % Initial lambdas
    end
    else        
    for k = 1:K
        betas(1,k,:)   = mvnrnd(mbar(k,:),eye(p)*1e2);                           % Initial beta
        lambdas(1,:)   = exp(sum(X.*squeeze(betas(1,sv,:)),2));                  % Initial lambdas
    end
    end
    
       
    for t = 1:T
    temp                    = diff([0 sort(rand(1,Y(t)))]);
    temp                    = [temp 1-sum(temp)+exprnd(lambdas(1,:))];
    tau(t,1:size(temp,2)) = temp;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gibbs step i %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for m = 2:niter
        
        betasi   =  squeeze(betas(m-1,:,:));
        Pi       =  squeeze(P(m-1,:,:));
        ipi      =  squeeze(ip(m-1,:))';
        filtsi   =  squeeze(filts(m-1,:))';
        lambdai  =  squeeze(lambdas(m-1,:))';
        lkl      =  zeros(T,1);
        clkli    =  zeros(T,K);
        filtpred =  zeros(T,K);
        
        %%%%%%%%%%%%%%%%%%%%% Draw Interarrival Times %%%%%%%%%%%%%%%%%%%%%

        st1t1 = ipi;
        
        
        for t = 1:T
            
        temp = diff([0 sort(rand(1,Y(t)))]);
        temp = [temp 1-sum(temp)+exprnd(lambdai(t))];
        n    = size(temp,2);
        tau(t,1:n) = temp;

        %%%%%%%%%%%%%%%%%%%%% Draw Mixture Indicators %%%%%%%%%%%%%%%%%%%%%
       
        indep             = X(t,:)*betasi(filtsi(t),:)';
        dep               = -log(temp)';
        [mu, sig , k]     = mixtures(dep,indep);
        Rm(t,1:n)         = mu';
        Rs(t,1:n)         = sig';
        
        %%%%%%%%%%%%%%%%%%%%% Rearrange the terms %%%%%%%%%%%%%%%%%%%%%%%%%
        
        ytilde  = dep-mu;
        xtilde  = repmat(X(t,:),n,1);
        
        Sigmat  = diag(sqrt(sig));
        
        %%%%%%%%%%%%%%%%%%%%% Draw the Hidden State %%%%%%%%%%%%%%%%%%%%%%%
        
        for k = 1:K
            
            xtildeb     = xtilde*betasi(k,:)';
            clklki      = sum(log(normpdf(ytilde,xtildeb,diag(Sigmat))));%log(mvnpdf(ytilde,xtildeb,Sigmat));%  
           
            clklki(isinf(clklki)) = -500;%median(clklki);
            clklki(isnan(clklki)) = -500;%nanmedian(clklki);
           
            clkli(t,k)  = clklki;

        end
          
        stt1          = Pi'*st1t1;                          % Prediction step
        clklmax       = max(clkli(t,:));
        stt           = stt1.*(exp(clkli(t,:)-clklmax)');   % Num updating
        nc            = sum(stt);                           % Normalizing const
        stt           = stt/nc;                             % Updating step
        lkl(t)        = log(nc)+clklmax;                    
        filtp(m,t,:)  = stt';
        st1t1         = stt;
        filtpred(t,:) = stt1';
        [~,maxpos]    = max(stt);
        filts(m,t)    = maxpos;

        Yt(t,1:n)     = ytilde';
        Xt(t,1:n,:)   = xtilde;
        
        end
        
        smooths(m,T)      = mnrnd(1,squeeze(filtp(m,T,:))')*[1:K]';
        smoothp(m,T,:)    = squeeze(filtp(m,T,:));
        
        for t=(T-1):-1:1
            smoothp(m,t,:)  = (Pi*(squeeze(smoothp(m,t+1,:))./filtpred(t+1,:)').*squeeze(filtp(m,t,:)))';
            p1              = (squeeze(filtp(m,t,:))).*Pi(:,smooths(m,t+1));
            p1              = p1/sum(p1);
            smooths(m,t)    = mnrnd(1,p1)*[1:K]';
        end
        clkl(m,:,:)   =  clkli;

        %%%%%%%%%%%%%%%%%%%%% Draw the Betas %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [BetaDraw]   = DrawBetas(Y,Yt,Xt,K,smooths(m,:),mbar,Mbar);
        lambda       = zeros(T,K);
        
        for k = 1:K
        lambda(:,k)  = exp(X*squeeze(BetaDraw(k,:))');
        end
        
%   Identification based on the intercept of the poisson regression

         chk     = BetaDraw(1,1) > BetaDraw(2,1);
%         
        for k = 2:K
            chk = BetaDraw(k-1,1) > BetaDraw(k,1);
        end
        
        %%%%%%%%%%%%%%%%%%%%% Draw the Transition Probabilities %%%%%%%%%%%
        
        [P_temp, ip_temp] = transitionprob(T,K,filts(m,:),P0);
                
        P(m,:,:)     = P_temp;
        ip(m,:)      = ip_temp;                
        betas(m,:,:) = BetaDraw;
        
  
        
for kk =1:K
        lambdas(m,find(filtsi==kk)) = lambda(find(filtsi==kk),1);
end        
        disp(['iteration ',num2str(m)]);
end

title = strcat('Results_', num2str(nn));

save(title) 
