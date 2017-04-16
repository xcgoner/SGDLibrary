function Problem = linear_regression(regularization)
% This file defines l2-regularized linear regression problem
%
% Inputs:
%       x_train     train data matrix of x of size nxd.
%       y_train     train data vector of y of size nx1.
%       x_test      test data matrix of x of size mxd.
%       y_test      test data vector of y of size mx1. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w) = 1/2 * (w' * x_i - y_i)^2.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Feb. 17, 2016
% Modified by H.Kasai on Oct. 25, 2016     
   
    Problem.name = @() 'linear_regression';    
    Problem.regularization = regularization;

    Problem.cost = @cost;
    function f = cost(x_train, y_train, w)

        f = sum((x_train * w - y_train).^2) / 2 / length(y_train) + regularization.cost(w);
        
    end

    Problem.grad = @grad;
    function g = grad(x_train, y_train, w, indices)

        residual = x_train(indices, :) * w - y_train(indices);
        g = (residual' * x_train(indices, :))' / length(indices);
        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(x_train, y_train, w)

        residual = x_train * w - y_train;
        g = (residual' * x_train)' / length(y_train);
        
    end

    Problem.prediction = @prediction;
    function p = prediction(x_test, w)
        p = x_test * w;        
    end

    Problem.rmse = @rmse;
    function e = rmse(y_test, y_pred)
        
        e = sqrt(sum((y_pred - y_test).^2)/ length(y_test));
        
    end

end

