function x = prox_l2(v, lambda)
% PROX_L1    The proximal operator of the l1 norm.
%
%   prox_l1(v,lambda) is the proximal operator of the l1 norm
%   with parameter lambda.

    x = max(0, (1 - lambda / norm(v))) * v;
end