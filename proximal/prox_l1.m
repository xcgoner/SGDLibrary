function regularization = prox_l1(lambda)
% PROX_L1    The proximal operator of the l1 norm.
%
%   prox_l1(v,lambda) is the proximal operator of the l1 norm
%   with parameter lambda.

    regularization.cost = @cost;
    function f = cost(v)
        f = lambda * norm(v, 1);
    end

    regularization.proximal = @proximal;
    function x = proximal(v, step_size)
        x = max(0, v - lambda * step_size) - max(0, -v - lambda * step_size);
    end

end