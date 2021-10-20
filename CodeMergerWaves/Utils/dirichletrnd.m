function draw = dirichletrnd(M,m,r,indE)
% PURPOSE: obtain n draws from m- variate dirichlet pdf
% -----------------------------------------------------
% USAGE: draw = draw_dirichlet(M,m,r,indE)
% where: 
% M    = number of draws
% m    = dimension of the dirichlet distribution
% r    = parameters of the distribution 
% indE = indicator telling whether draws are to be drawn 
%        from same distribution (indE~=0: in this case
%        r is a m x 1 vector), or to be drawn from different 
%        distribution (indE==0: in this case r is a (M x m) matrix
% NOTE: no dimension checks imposed , 09/06/04
% -----------------------------------------------------
% RETURNS: draw = (M x m) matrix with draws on different rows
%
% -----------------------------------------------------

if indE~=0
    rr=repmat(r',M,1);
else
    rr=r;
end
draw=chi2rnd(2*rr);
draw=draw./repmat(sum(draw,2),1,m);