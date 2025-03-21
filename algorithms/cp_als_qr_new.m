function [P,Uinit,output] = cp_als_qr_new(X,R,varargin)
    %CP_ALS_QR_NEW Compute a CP decomposition of any type of tensor.
    %
    %   M = cp_als_qr_new(X,R) computes an estimate of the best rank-R
    %   CP model of a tensor X using an alternating least-squares
    %   algorithm.  The input X can be a tensor, sptensor, ktensor, or
    %   ttensor. The result P is a ktensor.
    %
    %   M = cp_als_qr_new(X,R,'param',value,...) specifies optional parameters and
    %   values. Valid parameters and their default values are:
    %      'tol' - Tolerance on difference in fit {1.0e-4}
    %      'maxiters' - Maximum number of iterations {50}
    %      'dimorder' - Order to loop through dimensions {1:ndims(A)}
    %      'init' - Initial guess [{'random'}|'nvecs'|cell array]
    %      'printitn' - Print fit every n iterations; 0 for no printing {1}
    %
    %   [M,U0] = cp_als_qr_new(...) also returns the initial guess.
    %
    %   [M,U0,out] = cp_als_qr_new(...) also returns additional output that contains
    %   the input parameters.
    %
    %   Note: The "fit" is defined as 1 - norm(X-full(P))/norm(X) and is
    %   loosely the proportion of the data described by the CP model, i.e., a
    %   fit of 1 is perfect.
    %MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.
    
    % Adapted to use a QR (explicitely) to solve the LS problems instead of normal
    % equations, also includes timing of iteration parts, and outputs relative
    % error and fit. Used in 'EFFICIENT CP ROUNDING USING ALTERNATING LEAST SQUARES WITH QR DECOMPOSITION'
    
    
    
    %% Extract number of dimensions and norm of X.
    N = ndims(X);
    normX = norm(X);
    
    %% Set algorithm parameters from input or by using defaults
    params = inputParser;
    params.addParameter('tol',1e-4,@isscalar);
    params.addParameter('maxiters',50,@(x) isscalar(x) & x > 0);
    params.addParameter('dimorder',1:N,@(x) isequal(sort(x),1:N));
    params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
    params.addParameter('printitn',1,@isscalar);
    params.addParameter('errmethod','fast',@(x) ismember(x,{'fast','full','lowmem'}));
    params.parse(varargin{:});
    
    %% Copy from params object
    fitchangetol = params.Results.tol;
    maxiters = params.Results.maxiters;
    dimorder = params.Results.dimorder;
    init = params.Results.init;
    printitn = params.Results.printitn;
    errmethod = params.Results.errmethod;
    
    
    %% Error checking 
    
    %% Set up and error checking on initial guess for U.
    if iscell(init)
        Uinit = init;
        if numel(Uinit) ~= N
            error('OPTS.init does not have %d cells',N);
        end
        for n = dimorder(2:end)
            if ~isequal(size(Uinit{n}),[size(X,n) R])
                error('OPTS.init{%d} is the wrong size',n);
            end
        end
    else
        % Observe that we don't need to calculate an initial guess for the
        % first index in dimorder because that will be solved for in the first
        % inner iteration.
        if strcmp(init,'random')
            Uinit = cell(N,1);
            for n = dimorder(2:end)
                if isa(X(1),'single')
                    Uinit{n} = rand(size(X,n),R, 'single');
                else
                    Uinit{n} = rand(size(X,n),R);
                end
            end
        elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
            Uinit = cell(N,1);
            for n = dimorder(2:end)
                Uinit{n} = nvecs(X,n,R);
            end
        else
            error('The selected initialization method is not supported');
        end
    end
    
    %% Set up for iterations - initializing U and the fit.
    U = Uinit;
    fit = 0;
    
    if printitn>0
      fprintf('\nCP_ALS_QR_new (QR Implicit):\n');
    end
    
    
    %% Main Loop: Iterate until convergence
    
    %%% Changes for cp_als_qr start here: %%%
    
    %%% Initialize a cell array Qs and Rs to hold decompositions of factor matrices. %%%
    Qs = cell(N,1); %%% The Kronecker product of these tells us part of the Q of the Khatri-Rao product. %%%
    Rs = cell(N,1); %%% The Khatri-Rao product of these tells us the rest of Q and the R of the Khatri-Rao product. %%%
    
    total_t_ttm = 0; % TTM
    total_t_qrf = 0; % QR of factor matrices
    total_t_kr = 0; % Computing Q0
    total_t_q0 = 0; % Applying Q0
    total_t_back = 0;
    total_t_lamb = 0;
    total_t_err = 0;
    total_t_other = 0;
    
    %%% Compute economy-sized QR decomposition. %%%
    tic;
    for i = 1:N
        if ~isempty(U{i})
            [Qs{i}, Rs{i}] = qr(U{i},0); 
            
        end
    end
    total_t_qrf = total_t_qrf+toc;

    %%% TTM on all modes but the first one
    tic;
    Y = ttm(X,Qs,-1,'t');
    total_t_ttm = total_t_ttm + toc;
    
    for iter = 1:maxiters
        t_ttm = 0; % TTM
        t_qrf = 0; % QR of factor matrices
        t_kr = 0; % Computing Q0
        t_q0 = 0; % Applying Q0
        t_back = 0;
        t_lamb = 0;
        t_err = 0;
        
            
        fitold = fit;
            
        % Iterate over all N modes of the tensor
        for n = dimorder(1:end)
           
            if isa(Y,'ktensor')
                
                %%% For a ktensor: %%% 
                if n ~= N
                    R0 = Rs{N};
                    Z = Y{N}; 
                else
                    if isa(X(1),'single')
                        R0 = ones(1,R,'single');
                        Z = ones(1,size(Y{N},2), 'single'); 
                    else
                        R0 = ones(1,R);
                        Z = ones(1,size(Y{N},2)); 
                    end
                end
                %%% Compute pairwise QR decomposition
                %%% Update RHS in same way
                for k = N-1:-1:1
                    if k ~= n
                        tic; [Qp,R0] = qr(khatrirao(R0, Rs{k}),0); t = toc; t_kr = t_kr + t;
                        tic; Z = Qp' * khatrirao(Z,Y{k});  t = toc; t_q0 = t_q0 + t;
                    end
                end
                 tic;
                 Z = X.U{n} * Z';
                 t = toc;
                 t_ttm = t_ttm + t;
                
                %%% Calculate updated factor matrix by backsolving with R0' and Z. %%%
                tic;
                %Z = X.U{n} * Z';
                U{n} = Z / R0'; t = toc; t_back = t_back + t;  
            else
                %%% For any other tensor: %%%
                %%% TTM on all modes but n
                tic; Y = ttm(X,Qs,-n,'t'); t = toc; t_ttm = t_ttm + t;
    
                tic;  M = khatrirao(Rs{[1:n-1,n+1:N]},'r'); t = toc; t_kr = t_kr + t;
            
                %%% Compute the explicit QR factorization.
                tic;  [Q0,R0] = qr(M,0); t = toc; t_kr = t_kr + t;
                
                %%% Apply Q0 %%%
                tic; Z = tenmat(Y,n) * Q0; t = toc; t_q0 = t_q0 + t;
    
                %%% Calculate updated factor matrix by backsolving with R0' and Z. %%%
                %tic; U{n} = double(Z) / R0'; t = toc; t_back = t_back + t;
                tic; U{n} = Z.data / R0'; t = toc; t_back = t_back + t;
    
            end
                   
            % Normalize each vector to prevent singularities in coefmatrix
            tic;
            if iter == 1
                lambda = sqrt(sum(U{n}.^2,1))'; %2-norm
            else
                lambda = max( max(abs(U{n}),[],1), 1 )'; %max-norm
            end 
            
            Unew = bsxfun(@rdivide, U{n}, lambda'); t = toc; t_lamb = t_lamb + t;
            U{n} = Unew;
            
            %%% Recompute QR factorization for updated factor matrix. %%%
            tic; [Qs{n}, Rs{n}] = qr(U{n},0); t_qrf = toc;
            %%% Update TTM on mode n
            if isa(X,'ktensor')             
                tic; Y.U{n} = Qs{n}' * X.U{n}; t = toc; t_ttm = t_ttm + t;
            end
            
        end
    
        %%% Changes for cp_als_qr end here. %%%
            
        P = ktensor(lambda,U);
        
        tic;
        if normX == 0
           Rscaled = lambda.*R0';
           prod = U{dimorder(end)}*Rscaled;
           iprod = Z(:)'*prod(:);
           fit = norm(P)^2 - 2 * iprod;
        else
           switch errmethod
                case 'fast'
                    % fast inner product calculation
                    Rscaled = R0.*lambda';
                    prod = U{dimorder(end)}*Rscaled';
                    iprod = double(Z(:)')*double(prod(:));

                    % fast norm(P) calculation: < Lambda R0^T R0 Lambda, Rs{N}^T Rs{N} >
                    RscaledGram = double(Rscaled')*double(Rscaled);
                    RnGram = double(Rs{dimorder(end)}')* double(Rs{dimorder(end)});
                    normPsq = RscaledGram(:)'*RnGram(:);

                    normresidual = sqrt( abs(normX^2 + normPsq - 2 * iprod) );
                    
                    % normresidual = sqrt(abs( norm(X)^2 + norm(P)^2 - 2 * innerprod(X,P)));
                case 'full'
                    normresidual = norm(full(X) - full(P));
                case 'lowmem'
                    normresidual = normdiff(X,P); 
            end
            % class(normresidual)
            fit = 1 - (normresidual / normX); %fraction explained by model
            %%% Change this to just be relative error to see the error go down. %%%
            rel_err(iter,:) = normresidual / normX;
        end
        fitchange = abs(fitold - fit); t = toc; t_err = t_err + t;
            
        % Check for convergence
        if (iter > 1) && (fitchange < fitchangetol)
            flag = 0;
        else
            flag = 1;
        end
            
        %%% If the fit is NaN, just stop the process. %%%
        if isnan(fit)
            break;
        end
        
        if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
            fprintf('Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, fitchange);
        end
    
        % Check for convergence
        if (flag == 0)
            break;
        end     
        
        times(iter,:) = [t_ttm, t_qrf, t_kr, t_q0, t_back, t_lamb, t_err];
        
        total_t_ttm = total_t_ttm + t_ttm;
        total_t_qrf = total_t_qrf + t_qrf;
        total_t_kr = total_t_kr + t_kr;
        total_t_q0 = total_t_q0 + t_q0;
        total_t_back = total_t_back + t_back;
        total_t_lamb = total_t_lamb + t_lamb;
        total_t_err = total_t_err + t_err;
    end 

    
    
    %% Clean up final result
    % Arrange the final tensor so that the columns are normalized.
    P = arrange(P);
    % Fix the signs
    P = fixsigns(P);
    
    
    
    if printitn>0
        if normX == 0
            fit = norm(P)^2 - 2 * innerprod(X,P);
        else
            switch errmethod
                case 'fast'
                    normresidual = sqrt(abs( norm(X)^2 + norm(P)^2 - 2 * innerprod(X,P)));
                case 'full'
                    normresidual = norm(full(X) - full(P));
                case 'lowmem'
                    normresidual = normdiff(X,P);
            end
            fit = 1 - (normresidual / normX); %fraction explained by model
            rel_err(iter,:) = normresidual / normX;
        end
      fprintf(' Final f = %e \n', fit);
    end
    
    
    total_time = [total_t_ttm, total_t_qrf, total_t_kr, total_t_q0, total_t_back, total_t_lamb, total_t_err, total_t_other];
    output = struct;
    output.params = params.Results;
    output.iters = iter;
    output.relerr = rel_err; %%% Add a rel_err vector to output
    output.fit = fit;
    output.times = times;
    output.total_times = total_time;
    
    