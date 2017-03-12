function [w, infos] = dgdvr(problem, options)
% Stochastic average descent (SAG) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       options     options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       N. L. Roux, M. Schmidt, and F. R. Bach, 
%       "A stochastic gradient method with an exponential convergence rate for finite training sets,"
%       NIPS, 2012.
%    
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Jan. 12, 2017


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();

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

    if ~isfield(options, 'batch_size')
        batch_size = 10;
    else
        batch_size = options.batch_size;
    end
    num_of_bachces = floor(n / batch_size);    
    
    if ~isfield(options, 'prob')
        prob_max = 0.6;
    else
        prob_max = options.prob;
    end  
    
    if ~isfield(options, 'prob_decay')
        prob_decay_init = prob_max;
        prob_decay_rate = 0;
    else
        prob_decay_init = options.prob_decay.init;
        prob_decay_rate = options.prob_decay.rate;
    end
    
    if ~isfield(options, 'max_epoch')
        max_epoch = 100;
    else
        max_epoch = options.max_epoch;
    end 
    
    if ~isfield(options, 'w_init')
        w = randn(d,1);
    else
        w = options.w_init;
    end 
    
    if ~isfield(options, 'sub_mode')
        sub_mode = 'DGD-NOVR';
    else
        sub_mode = options.sub_mode;
    end     
    
    if ~isfield(options, 'f_opt')
        f_opt = -Inf;
    else
        f_opt = options.f_opt;
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
    
    
    % initialize
    iter = 0;
    epoch = 0;
    grad_calc_count = 0;

    % store first infos
    clear infos;
    infos.iter = epoch;
    infos.time = 0;    
    infos.grad_calc_count = grad_calc_count;
    f_val = problem.cost(w);
    optgap = f_val - f_opt;
    infos.optgap = optgap;
    infos.gnorm = norm(problem.full_grad(w));        
    infos.cost = f_val;
    if store_w
        infos.w = w;       
    end      

    % set start time
    start_time = tic();
    
    % permute samples (ToDo)
    perm_idx = 1:n;     
    
    prob = prob_decay_init;

    % main loop
    while (optgap > tol_optgap) && (epoch < max_epoch)

        % update step-size
        if strcmp(step_alg, 'decay')
            step = step_init / (1 + step_init * lambda * iter);
        end
        
        % batches to be update
        if epoch == 1
            batch_index = 1:num_of_bachces;
        else
            batch_index = find(binornd(1, prob, 1, num_of_bachces));
        end
        
        grad_aggregate = zeros(d, 1);
        for jj=1:length(batch_index)
            j = batch_index(jj);

            % calculate gradient
            start_index = (j-1) * batch_size + 1;
            indice_j = perm_idx(start_index:start_index+batch_size-1);
            grad = problem.grad(w, indice_j);
            
            grad_aggregate = grad_aggregate + grad;

        end
        
        % update step-size
        if strcmp(step_alg, 'decay')
            step = step_init / (1 + step_init * lambda * iter);
        end
        
        % update w
        w = w - step * grad_aggregate / (length(batch_index) * batch_size);
        
        if epoch > 1 && infos.cost(end) > infos.cost(end-1)
            prob = min(prob_max, prob * (1+prob_decay_rate));
        end
        
        iter = iter + 1;
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + length(batch_index) * batch_size;        
        % update epoch
        epoch = epoch + 1;
        % calculate optgap
        f_val = problem.cost(w);
        optgap = f_val - f_opt; 
        % calculate norm of full gradient
        gnorm = norm(problem.full_grad(w) / problem.samples());      

        % store infos
        infos.iter = [infos.iter epoch];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        infos.optgap = [infos.optgap optgap];
        infos.cost = [infos.cost f_val];
        infos.gnorm = [infos.gnorm gnorm];            
        if store_w
            infos.w = [infos.w w];         
        end          

        % display infos
        if verbose > 0
            fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e, prob = %.6e\n', sub_mode, epoch, f_val, optgap, prob);
        end
    end
    
    if optgap < tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', tol_optgap);
    elseif epoch == max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', max_epoch);
    end    
end
