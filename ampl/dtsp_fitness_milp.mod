# DTSP, fitness evaluation by MILP
param T := 10;             # time horizon (hours), initial time stage = 0, time period [0,T]
param n;                   # total number of ships visiting the work area during [0,T]
let n := 63;               # total number of ships visiting the work area during [0,T]
param w := 5/60;           # time slot (hours)
param m := floor(T/w+0.1); # number of time slots in [0,T]
param x{kk in 0..190, ii in 0..63}; # x-coordinate (km) of ship i in slot k; i=0 is the harbor
param y{kk in 0..190, ii in 0..63}; # y-coordinate (km) of ship i in slot k; i=0 is the harbor 
                           # for i>0, coordinates (x[k,i],y[k,i])=(0,0) means that ship i is outside the work area in slot k
param first{ii in 1..63};  # first slot when ship i in in the work area
param  last{ii in 1..63};  #  last slot when ship i in in the work area
param p := 3/60;           # service time (hours); NOTE: we requier p<w
param v := 46.3;           # speed (km/h) of the service vessel
param j;                   # auxiliary param

param alpha;               # input for fitness enaluation: number of ships to be visited during [0,T]; for a single optimixation fix alpha, 0<alpha<=n
param seq{si in 1..alpha}; # input for fitness enaluation: sequence of ships to be visited; seq defines one member of the population
set J := 1..alpha;
set S0 := 1..m;
set S{ii in J} := first[seq[ii]]..last[seq[ii]];

var z{ii in J, kk in S[ii], ll in S[ii+1]: ii<alpha and ll>kk} binary; # move from (ii,kk) to (ii+1,ll)
var z0{ll in S[1]}     binary;             # move from harbor to first ship seq[1] in slot ll
var zn{kk in S[alpha]} binary;             # move from last ship seq[alpha] in slot kk to harbor

var s{kk in S0} >=(if kk=m then 0 else (kk-1)*w) <=kk*w;  # service start of slot kk

minimize obj:  sum{ii in J, kk in S[ii], ll in S[ii+1]: ii<alpha and ll>kk} sqrt((x[kk,seq[ii]]-x[ll,seq[ii+1]])^2+(y[kk,seq[ii]]-y[ll,seq[ii+1]])^2)*z[ii,kk,ll] 
             + sum{ll in S[1]: ll>0}     sqrt(x[ll,seq[1]]^2    +y[ll,seq[1]]^2)*z0[ll] 
             + sum{kk in S[alpha]: kk>0} sqrt(x[kk,seq[alpha]]^2+y[kk,seq[alpha]]^2)*zn[kk];

  init: sum{ll in S[1]}     z0[ll] = 1;

finish: sum{kk in S[alpha]} zn[kk] = 1;

inrerm{ii in J, kk in S[ii]}: (if ii=1 then z0[kk] else sum{ll in S[ii-1]: kk>ll}     z[ii-1,ll,kk]) 
                            = (if ii=alpha then zn[kk] else sum{ll in S[ii+1]: ll>kk} z[ii,kk,ll]);

start{ii in J, kk in S[ii], ll in S[ii+1]: ii<alpha and  ll>kk}: 
       s[ll] >= s[kk] + (p + sqrt((x[kk,seq[ii]]-x[ll,seq[ii+1]])^2+(y[kk,seq[ii]]-y[ll,seq[ii+1]])^2)/v)*z[ii,kk,ll]; 

start0{ll in S[1]}: s[ll] >= (sqrt(x[ll,seq[1]]^2+y[ll,seq[1]]^2)/v)*z0[ll];

startn{kk in S[alpha]}: s[m] >= s[kk] + (p+sqrt(x[kk,seq[alpha]]^2+y[kk,seq[alpha]]^2)/v)*zn[kk];

param shipn{ii in 1..63}; # original ship numbers for new ship numbers ii
param shipo{ii in 1..63}; # new ship numbers for original ship numbers ii (if ii is not excluded)
#   in files x.dat and y.dat the harbor coordinates are translated to (0,0); for i>0, x=0 or y=0 'mean out of work area'
  # data x.dat; data y.dat;  # data x.txt; data y.txt; # excel data from Alaleh
  let{ii in 1..n} first[ii] := min(m,min{kk in 0..m: x[kk,ii]!=0 and y[kk,ii]!=0} kk); # find first slot for i in the work area
  let{ii in 1..n}  last[ii] := max(0,max{kk in 0..m: x[kk,ii]!=0 and y[kk,ii]!=0} kk); # find  last slot for i in the work area
  let j := 0;
  for{jj in 1..n: first[jj]<=last[jj]}{
    let j := j + 1;
    let{ii in 0..m} x[ii,j]:= x[ii,jj];
    let{ii in 0..m} y[ii,j]:= y[ii,jj];
    let first[j]:= first[jj];
    let  last[j]:=  last[jj];
    let shipn[j]  := jj;
    let shipo[jj] := j;
  };
  let n := j;
  display n; # number of eligible ships
  display m; # number of time slots

############## begin test case: note the renumbering of ships after schrinking ###
# for fittness evaluation, enter here values alpha and seq(*):
  # let alpha := 5;      # test case, number of ships to be visited during [0,T];
  # let seq[1]  := shipo[56]; # temporary sequence for testing
  # let seq[2]  := shipo[26]; # temporary sequence for testing
  # let seq[3]  := shipo[33]; # temporary sequence for testing
  # let seq[4]  := shipo[ 8]; # temporary sequence for testing
  # let seq[5]  := shipo[12]; # temporary sequence for testing
### end test case ###

# MIP for fittness evaluation
option solver mosek;
option mosek_options' outlev=0 MSK_IPAR_INTPNT_MAX_ITERATIONS=500 MSK_DPAR_INTPNT_NL_TOL_REL_GAP=1e-8 MSK_DPAR_OPTIMIZER_MAX_TIME=3600';
solve;
############## end of test case: note the renumbering of ships after schrinking ###
