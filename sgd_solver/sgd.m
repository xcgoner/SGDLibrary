function [w, infos] = sgd(problem, options)
% Stochastic gradient descent algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Jan. 12, 2017

    % extract options
    if ~isfield(options, 'step_init')
        step_init = 0.1;
    else
        step_init = options.step_init;
    end
    step = step_init;    
    
    if ~isfield(options, 'step_alg')
        step_alg = 'fix';
    else
        if strcmp(options.step_alg, 'decay')
            step_alg = 'decay';
        elseif strcmp(options.step_alg, 'fix')
            step_alg = 'fix';
        else
            step_alg = 'decay';
        end
    end  
    
    if ~isfield(options, 'lambda')
        lambda = 0.1;
    else
        lambda = options.lambda;
    end 
    
    if ~isfield(options, 'tol_optgap')
        tol_optgap = 1.0e-12;
    else
        tol_optgap = options.tol_optgap;
    end         
    
    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'max_iter')
        max_iter = 10;
    else
        max_iter = options.max_iter;
    end 
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end 
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
    end 
    
    if ~isfield(options, 'permute_on')
        permute_on = 1;
    else
        permute_on = options.permute_on;
    end     
    
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    if ~isfield(options, 'store_w')
        store_w = false;
    else
        store_w = options.store_w;
    end      
    
    if ~isfield(options, 'reader')
        reader = false;
    else
        reader = options.reader;
    end   
    
    if ~isfield(problem, 'test_data')
        test_data = false;
    else
        test_data = true;
    end
    
    if ~isfield(problem, 'regularization')
        regularization = false;
    else
        regularization = true;
        disp('regularized!');
    end
    
    
    % initialize
    iter = 0;
    epoch = 0;
    grad_calc_count = 0;

    % store first infos
    clear infos;
    infos.iter = epoch;
    infos.time = 0;    
    infos.grad_calc_count = grad_calc_count;
    f_val = problem.rmse(problem.test_data.y, problem.prediction(problem.test_data.X, w))
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    infos.cost = f_val;
    if store_w
        infos.w = w;       
    end      

    % set start time
    start_time = tic();

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)
        
        batch = reader.nextBatch();
        
        f_val = problem.rmse(problem.test_data.y, problem.prediction(problem.test_data.X, w));
        w_tmp = w;
        % multiple steps for one mini-batch
        for i = 1:max_iter
            grad = problem.full_grad(batch.X, batch.y, w_tmp);
            w_tmp = w_tmp - step * grad;
            if regularization == true
                % search proximal
                f_val_tmp = problem.rmse(problem.test_data.y, problem.prediction(problem.test_data.X, w_tmp))
                w_prox = w_tmp;
                for j = 1:6
                    w_tmp_1 = problem.regularization.proximal(w_tmp, step * (10^j));
                    f_val_tmp_1 = problem.rmse(problem.test_data.y, problem.prediction(problem.test_data.X, w_tmp_1))
                    if f_val_tmp_1 < f_val_tmp
                        f_val_tmp = f_val_tmp_1;
                        w_prox = w_tmp_1;
                    end
                end
                w_tmp = w_prox;
            end
            f_val_tmp = problem.rmse(problem.test_data.y, problem.prediction(problem.test_data.X, w_tmp))
            if f_val_tmp < f_val
                f_val = f_val_tmp;
                w = w_tmp;
            else 
                break;
            end
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + length(batch.y);        
        % update epoch
        epoch = epoch + 1;
        % calculate optimality gap
        optgap = f_val - f_opt;        

        % store infos
        infos.iter = [infos.iter epoch];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        infos.optgap = [infos.optgap optgap];
        infos.cost = [infos.cost f_val];     
        if store_w
            infos.w = [infos.w w];         
        end           

        % display infos
        if verbose > 0
            fprintf('SGD: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
        end

    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end
    
end
