% Euxhen Hasanaj
% Soft-SVM Dual implementation with Laplace kernel
%
%We are minimizing over alph the following
%   1/2*sum(alph_i*alph_j*y_i*y_j*k(x_i, x_j))-sum(alph_i)
% subject to 0<=alph_i<=C for all i and sum(alph_i*y_i)=0.
%
% We can rewrite the problem as
%   1/2*alph'*((y*y').*K)*alph-ones(n,1)'*alph
% subject to zeros(n,1) <= alph <= C * ones(n, 1)
% and [y'; zeros(n-1, n)] * alph = zeros(n,1).

function [K, alph, b, fval] = svmlaplace(x, y, lambda, C)

k = @(v1, v2) exp(-lambda * norm(v1 - v2, 1));  % Laplace kernel
n = length(x);

% This will be the Gram matrix
% K(i,j) = k(x_i, x_j)
K = []; 
for i = [1:n]
    for j = [i:n]
        K(i,j) = k(x(i, :), x(j, :));
        K(j,i) = K(i,j); % Using symmetry
    end
end

Y = y * y'; % Needed since we have y_i * y_j in the problem

Aeq = [y'; zeros(n-1, n)];  % since sum(y_i * alph_i) = 0
beq = zeros(n,1);

LB = zeros(n,1);  % since alph >= 0
UB = C*ones(n,1); % since alph <= C

[alph, fval] = quadprog(K.*Y, -ones(n,1), [], [], Aeq, beq, LB, UB);

% Now we find the value of b by picking any support vector
threshold = 1e-4; % Some threshold for determining a support vector
assert(C > threshold);
index = find(alph > threshold, 1);
b = y(index) - (alph .* y)' * K(:, index);

end
