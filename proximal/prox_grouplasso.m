function regularization = prox_grouplasso(groupsize, lambda)
% PROX_L1    The proximal operator of the group lasso norm.
%
%   prox_grouplasso(v, step_size) is the proximal operator of the group lasso norm
%   with parameter lambda.

    regularization.cost = @cost;
    function f = cost(v)
        f = 0;
        n = length(v) / groupsize;
        for i = 1:n
            f = f + norm(v(((i-1)*groupsize+1):i*groupsize), 2);
        end
        f = f * lambda;
    end

    regularization.proximal = @proximal;
    function x = proximal(v, step_size)
        c = lambda * step_size;
        n = length(v) / groupsize;
        x = zeros(size(v));
        for i = 1:n
            idx1 = (i-1)*groupsize+1;
            idx2 = i*groupsize;
            factor = max( 0, 1 - c / norm( v(idx1:idx2), 2 ) );
            x(idx1:idx2) = factor * v(idx1:idx2);
        end
    end

end