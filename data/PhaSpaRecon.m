function xn = PhaSpaRecon(s,tau,m)
% Phase space reconstruction for chaotic time series
% [xn] = PhaSpaRecon(s,tau,m)
% Input:
%       s        original time series
%       tau      time delay
%       m        embedding dimension
% Output:
%       xn       reconstructed phase space (each column is one state vector)

len = length(s);
if (len - (m-1)*tau < 1)
    disp('err: delay time or embedding dimension is too large!')
    xn = [];
else
    xn = zeros(m, len - (m-1)*tau);
    for i = 1:m
        xn(i,:) = s(1+(i-1)*tau : len-(m-i)*tau);
    end
end