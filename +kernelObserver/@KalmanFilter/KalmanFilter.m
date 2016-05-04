%============================= KalmanFilter ===============================
%
%  Implements a simple Kalman filter that is used by KernelObserver class.  
% 
%  Inputs:
%    fmap_type  - string indication feature map: 
%                 {"RBFNetwork", "RandomKitchenSinks"}.
%    map_struct - struct with mapping parameters
%
%  Outputs:
%               -see functions
%
%============================= KalmanFilter ===============================
%
%  Name:      KalmanFilter.m
%
%  Author:    Hassan A. Kingravi
%
%  Created:   2016/05/02
%  Modified:  2016/05/02
%
%============================= KalmanFilter ===============================
classdef KalmanFilter < handle 
  % class properties   
  properties (Access = public)        
  end
    
  % hidden variables 
  properties (Access = protected)    
    % internal matrices
    P_k_prev = [];    
    Q        = [];
    R        = [];
    
    % dynamics matrices
    A        = [];  % dynamics
    C        = [];  % measurement matrix
    nstates  = [];  % number of states
    
    m_prev   = [];  % state vector
  end
  
  % class methods 
  methods    
    function obj = KalmanFilter(P_init, Q, R)
      %  Constructor for KalmanFilter: initialize filter parameters.
      %
      %  Inputs:
      %    P_init     - initial m x m error covariance matrix
      %    Q          - m x m process noise covariance matrix
      %    R          - n x n measurement noise covariance matrix
      %
      %  Outputs:
      %    -none
      obj.P_k_prev = P_init;
      obj.Q = Q;
      obj.R = R;
    end
    
    function fit(obj, A, C, m_init)
      %  'Fit' filter with dynamics and measurement operators, and the
      %  initial state. 
      %
      %  Inputs:
      %    A          - dynamics operator: m x m matrix
      %    C          - measurement operator: n x m matrix
      %    meas_prev  - current measurement vector
      %
      %  Outputs:
      %    -none
      obj.A = A;
      obj.C = C;
      obj.m_prev = m_init;
      obj.nstates = size(obj.A, 1);
    end
                
    function [m_k, P_k] = predict(obj, meas_curr)
      %  Given a vector of current measurements, filter and correct current
      %  state. 
      %
      %  Inputs:
      %    meas_curr  - current measurement vector
      %
      %  Outputs:
      %    m_k        - a posteriori m x 1 state estimate
      %    P_k        - a posteriori m x m error covariance matrix
      %
      m_k_pred = obj.A*obj.m_prev;
      P_k_pred = obj.A*obj.P_k_prev*obj.A' + obj.Q;
      v_k = meas_curr - obj.C*m_k_pred;
      S_k = obj.C*P_k_pred*obj.C' + obj.R;
      C_k = P_k_pred*obj.C'/S_k;
      m_k = m_k_pred + C_k*v_k;
      P_k = (eye(obj.nstates) - C_k*obj.C)*P_k_pred;
      
      obj.m_prev = m_k;
      obj.P_k_prev = P_k;
    end
    
    function mval = get(obj,mfield)
      % Get a requested member variable.
      %
      switch(mfield)
        case {'A'}
          mval = obj.A;
        case {'C'}
          mval = obj.C;
        case {'meas_prev'}
          mval = obj.meas_prev;
        case {'P_k_prev'}
          mval = obj.P_k_prev;          
        case {'Q'}
          mval = obj.Q;                    
        case {'R'}
          mval = obj.R;                    
        otherwise
          disp('wrong variable name')
      end
    end
    % end methods
  end 
end