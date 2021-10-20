function draw = draw_multinom(p,T,m,indE)
% PURPOSE: filtering on mixture models 
% -----------------------------------------------------
% USAGE: 
% where: 
% NOTE: no dimension checks imposed , 08/06/04
% -----------------------------------------------------
% RETURNS: 
% 
%
% -----------------------------------------------------

if indE==0
   pp=cumsum(p,2);    
else
    pp=repmat((cumsum(p))',T,1);
end
[d1,d2]=max(repmat(rand(T,1),1,m)<pp,[],2);
draw=d2;