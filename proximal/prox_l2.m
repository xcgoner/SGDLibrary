function regularization = prox_l2(lambda)
% PROX_L1    The proximal operator of the l1 norm.
%
%   prox_l1(v,lambda) is the proximal operator of the l1 norm
%   with parameter lambda.

    regularization.cost = @cost;
    function f = cost(v)
        f = lambda * norm(v);
    end

    regularization.proximal = @proximal;
    function x = proximal(v, step_size)
        x = max(0, (1 - lambda * step_size / norm(v))) * v;
    end
end