%======================= time_varying_uncertainty =========================
%  
%  This function implements the time varying uncertainty by changing the
%  weight vector Wstar given its previous values and the current time. 
%
%  There are currently three options, given by the variable scheme.
%    - smooth1     - vary weights smoothly, with periodic functions
%    - smooth2     - vary weights smoothly, with periodic functions (unstable)
%    - switch      - switch between various smooth schemes
%    - nonperiodic - have nonperiodic weights
% 
%  Reference(s): 
%    none
% 
%  Inputs:
%    Wstar	    - 1 x 5 vector containing previous weights
%    t          - 1 x 1 scalar current time
%    
%  Outputs:
%    Wstar	    - 1 x 5 vector updated vector
%
%======================= time_varying_uncertainty =========================
%  Name:	time_varying_uncertainty.m
%
%  Author(s): Hassan A. Kingravi
%
%  Created:  2012/08/09
%  Modified: 2014/05/14
%======================= time_varying_uncertainty =========================
function [Wstar] = time_varying_uncertainty(Wstar, t, scheme)
  
  if strcmp(scheme,'smooth1')
    Wstar(1) = 0.999*Wstar(1) + 0.5*sin(t)*Wstar(2) + 0.5*sin(t);
    Wstar(2) = 0.3*cos(1.1*t) + 0.1*Wstar(2);
    Wstar(3) = max(min(0.1*cos(1.1*t) + Wstar(3),2),-2);
    Wstar(4) = sin(Wstar(5)) + Wstar(4)*max(min(0.1*cos(1.1*t) + 0.3*sin(25*t),2),-2);
    Wstar(5) = 0.1*cos(2.2*t) + max(min(0.1*cos(1.1*t)*Wstar(1) + Wstar(5),2),-2);      
  elseif strcmp(scheme,'smooth2') % switch between a series of weights    
    Wstar(1) = Wstar(1) + 0.1*sin(t);
    Wstar(2) = cos(1.1*t) + Wstar(2);
    Wstar(3) = max(min(0.1*cos(1.1*t) + Wstar(3),2),-2);
    Wstar(4) = max(min(0.1*cos(1.1*t) + 0.3*sin(25*t),2),-2);
    Wstar(5) = cos(2.2*t) + 0.01*Wstar(5);          
  elseif strcmp(scheme,'smooth3')
    Wstar(1) = 0.1*Wstar(1) + 0.2*sin(t)*Wstar(2) + 0.2*sin(t);
    Wstar(2) = 0.4*cos(1.1*t) + 0.5*Wstar(2);
    Wstar(3) = max(min(0.1*cos(1.1*t) + Wstar(3),2),-2);
    Wstar(4) = 0.3*(sin(Wstar(5)) + Wstar(4)*max(min(0.1*cos(1.1*t) + 0.3*sin(25*t),2),-2));
    Wstar(5) = 0.8*cos(2.2*t) + max(min(0.1*cos(1.1*t)*Wstar(1) + Wstar(5),2),-2);
  elseif strcmp(scheme,'smooth4')
    Wstar(1) = 0.7*Wstar(1) + 0.2*sin(t)*Wstar(2) + 0.3*sin(t);
    Wstar(2) = 0.3*cos(1.1*t) + 0.1*Wstar(2);
    Wstar(3) = max(min(0.1*cos(1.1*t) + Wstar(3),2),-2);
    Wstar(4) = sin(Wstar(5)) + Wstar(4)*max(min(0.1*cos(1.1*t) + 0.3*sin(25*t),2),-2);
    Wstar(5) = 0.1*cos(2.2*t) + max(min(0.1*cos(1.1*t)*Wstar(1) + Wstar(5),2),-2);      
  elseif strcmp(scheme, 'switching') % switch between a series of weights
    if t < 5   
      Wstar(1) = 0.999*Wstar(1) + 0.001*sin(t);
      Wstar(2) = 0.1*cos(1.1*t) + 0*Wstar(2);
      Wstar(3) = max(min(0.1*cos(1.1*t) + Wstar(3),2),-2);
      Wstar(4) = max(min(0.1*cos(1.1*t) + 0.3*sin(25*t),2),-2);
      Wstar(5) = 0.01*cos(2.2*t) + 0.01*Wstar(5);              
    elseif t>5 && t < 10
      Wstar(1) = 0.9*Wstar(1) + 0.001*cos(t);
      Wstar(2) = 0.1*cos(0.5*t) + 0*Wstar(2);
      Wstar(3) = max(min(1*cos(1.2*t) + Wstar(3),1.5),-1.5);
      Wstar(4) = max(min(0.1*cos(1.1*t) + 0.3*sin(30*t),1.2),-1.2);
      Wstar(5) = 0.1*cos(2.2*t) + 0.01*Wstar(5);                      
    else
      Wstar(1) = 0.9*Wstar(1) + 0.01*cos(t);
      Wstar(2) = 0.1*cos(1.0*t) + 0*Wstar(2);
      Wstar(3) = max(min(0.4*cos(1.2*t) + Wstar(3),1.0),-1.0);
      Wstar(4) = max(min(0.2*cos(1.1*t) + 0.3*sin(28*t),2),-2);
      Wstar(5) = 0.1*cos(2.0*t) + 0.01*Wstar(5);                              
    end     
  elseif strcmp(scheme, 'switching2') % switch between a series of weights
    if t < 5
      Wstar(1) = 0.999*Wstar(1) + 0.001*sin(t);
      Wstar(2) = 0.1*cos(1.1*t) + 0*Wstar(2);
      Wstar(3) = max(min(0.1*cos(1.1*t) + Wstar(3),2),-2);
      Wstar(4) = max(min(0.1*cos(1.1*t)*Wstar(1) + 0.3*sin(25*t),2),-2);
      Wstar(5) = 0.01*cos(2.2*t)*Wstar(2) + 0.01*Wstar(5);
    elseif t>5 && t < 10
      Wstar(1) = 0.9*Wstar(1) + 0.001*cos(t);
      Wstar(2) = 0.1*cos(0.5*t*Wstar(1)) + 0*Wstar(2);
      Wstar(3) = max(min(1*cos(1.2*t) + Wstar(3),1.5),-1.5);
      Wstar(4) = max(min(0.1*cos(1.1*t*Wstar(3)) + 0.3*sin(30*t),1.2),-1.2);
      Wstar(5) = 0.1*cos(2.2*t)*Wstar(2) + 0.01*Wstar(5);
    else
      Wstar(1) = 0.9*Wstar(1)*Wstar(2) + 0.01*cos(t);
      Wstar(2) = 0.1*cos(1.0*t) + Wstar(3);
      Wstar(3) = max(min(0.4*cos(1.2*t) + Wstar(3),1.0),-1.0);
      Wstar(4) = Wstar(5)*max(min(0.2*cos(1.1*t) + 0.3*sin(28*t),2),-2);
      Wstar(5) = 0.1*cos(2.0*t*Wstar(1)) + 0.01*Wstar(5);
    end
  elseif strcmp(scheme, 'fast_switching')
    if t < 5   
        Wstar = [0.3 0.4 0.1 0.4 -0.45 0.0214]'; 
    elseif t>15 && t < 35
      Wstar = [-0.2 0.1 0.3 -0.14 0.1 -0.214]';
    else
    Wstar = [0.9 0.5 -0.233 0.3 -0.2 0.04]';
    end  
      
  end

end

