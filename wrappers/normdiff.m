function err = normdiff(A,B)
% computes exact relative error between two ktensors A,B in a
% memory-efficient manner

d = length(B.U);
dims = size(A);
halves = floor(dims/2);

Asub = A;
Bsub = B;

sqerr = 0;

for i = 0:2^(d)-1
    for j = 1:d
        if bitget(i,j)
            Asub.U{j} = A.U{j}(1:halves(j),:);
            Bsub.U{j} = B.U{j}(1:halves(j),:);
        else
            Asub.U{j} = A.U{j}(halves(j)+1:end,:);
            Bsub.U{j} = B.U{j}(halves(j)+1:end,:);
        end
    end
    sqerr = sqerr + (norm(full(Asub)-full(Bsub)))^2;
end

err = sqrt(sqerr);
end


% %% parallel function 
% function err = normdiff(A, B)
% 
%     tic;
%     if nargin < 3
%         dparts = 15;
%     end
% 
%     if nargin < 4
%         mode = 'parallel';
%     end
% 
%     % Validate input mode
%     if ~ismember(mode, {'sequential', 'parallel'})
%         error('Invalid mode. Choose either ''sequential'' or ''parallel''.');
%     end
% 
% 
% 
% 
%     d = length(B.U);  % Number of dimensions (modes) of the tensor
%     dims = size(A);  % Size of each mode (dimension) from A's factors
%     dpart_sizes = floor(dims / dparts);  
% 
%     % Initialize variables
%     sqerr = 0;  % Accumulator for squared error
%     numCombinations = dparts^d;  % Total number of sub-tensor combinations (since 12 possibilities for each dimension)
% 
%     % Preallocate array for parallel accumulation
%     sqerrArray = zeros(numCombinations, 1);
% 
%     index_tuple = generateTuples(d,dparts);
%     % Parallel loop over all combinations of sub-tensors
% 
%     if strcmp(mode, 'sequential')
%         for i = 1:numCombinations
%             % Local variables for each worker
%             Asub = A;
%             Bsub = B;
% 
%             % Adjust sub-tensors based on the division into 12 slices
%             for j = 1:d
%                 if index_tuple(i,j)== dparts
%                     startIdx = (index_tuple(i,j)-1)*dpart_sizes(j)+1;
%                     Asub.U{j} = A.U{j}(startIdx:end, :);
%                     Bsub.U{j} = B.U{j}(startIdx:end, :);
%                 else
%                     startIdx = (index_tuple(i,j)-1)*dpart_sizes(j) +1;
%                     endIdx = index_tuple(i,j)*dpart_sizes(j);
%                     Asub.U{j} = A.U{j}(startIdx:endIdx, :);
%                     Bsub.U{j} = B.U{j}(startIdx:endIdx, :);
%                 end
% 
%             end
%             sqerrArray(i) = (norm(full(Asub) - full(Bsub)))^2;
%          end
%     elseif strcmp(mode, 'parallel')
%         parpool('local',30);
%         parfor i = 1:numCombinations
% 
%             % Local variables for each worker
%             Asub = A;
%             Bsub = B;
% 
%             % Adjust sub-tensors based on the division into 12 slices
%             for j = 1:d
%                 if index_tuple(i,j)== dparts
%                     startIdx = (index_tuple(i,j)-1)*dpart_sizes(j)+1;
%                     Asub.U{j} = A.U{j}(startIdx:end, :);
%                     Bsub.U{j} = B.U{j}(startIdx:end, :);
%                 else
%                     startIdx = (index_tuple(i,j)-1)*dpart_sizes(j) +1;
%                     endIdx = index_tuple(i,j)*dpart_sizes(j);
%                     Asub.U{j} = A.U{j}(startIdx:endIdx, :);
%                     Bsub.U{j} = B.U{j}(startIdx:endIdx, :);
%                 end
% 
%             end
%             sqerrArray(i) = (norm(full(Asub) - full(Bsub)))^2;
%         end
%        delete(gcp());
%     end
% 
%     % Aggregate the squared errors from all workers
%     sqerr = sum(sqerrArray);
% 
%     % Compute the final relative error
%     err = sqrt(sqerr);
%     toc
%     maxNumCompThreads(1);
% end
% 
% function tuples = generateTuples(n, d)
%     % Generate all n-tuples where each index takes values between 1 and d.
% 
%     % Create a cell array to hold grid vectors
%     grids = cell(1, n);
% 
%     % Generate the grid vectors
%     [grids{:}] = ndgrid(1:d);
% 
%     % Concatenate the grids into an n-tuples matrix
%     tuples = cell2mat(cellfun(@(x) x(:), grids, 'UniformOutput', false));
% end