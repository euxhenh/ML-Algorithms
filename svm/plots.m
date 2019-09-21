% Euxhen Hasanaj
% Helper function for plotting

function t = plots()

S = load('data.txt');
x = S(:, 1:2);
y = S(:, 3);

n = length(x);
% Counters
id = 1;
ii = 1;
jj = 1;

for lambda = [0.1, 1, 10, 20]
    for C = [0.01, 0.1, 0.5, 1]
        % Run Soft-SVM
        [K, alph, b, fval] = svmlaplace(x, y, lambda, C);
        fvals(ii, jj) = fval;

        ylist = [];
        counter = 1;
        % Calculate prediction for every data point
        for i = [1:n]
            yi = (alph .* y)' * K(:, i) + b;
            ylist(counter) = sign(yi);
            counter = counter + 1;
        end

        accuracy = sum(y == ylist') / n;
        acc(ii, jj) = accuracy;
        % Plotting
        subplot(4, 4, id);
        colors = [ylist == 1]' * [0 0 1]; % blue
        colors = colors + [ylist == -1]' * [0 1 0]; % red
        scatter(x(:, 1), x(:, 2), 4, colors);
        title(sprintf('lambda=%.1f, C=%.2f\nacc=%.2f\nOptimal value=%.2f', lambda, C, accuracy, fval));

        id = id + 1;
        sprintf('Finished learning lambda=%f, C=%f', lambda, C)
        jj = jj + 1;
    end
    ii = ii + 1;
    jj = 1;
end
fprintf('Accuracy table')
disp(acc)
fprintf('Optimal values table')
disp(fvals)

end
