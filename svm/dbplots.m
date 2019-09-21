% Euxhen Hasanaj
% 10715, Fall 2019
% Helper function for plotting

function t = dbplots()

S = load('data.txt');
x = S(:, 1:2);
y = S(:, 3);

n = length(x);
% Counters
id = 1;

for lambda = [0.1, 1, 10, 20]
    for C = [0.01, 0.1, 0.5, 1]
        % Run Soft-SVM
        [K, alph, b, fval] = svmlaplace(x, y, lambda, C);
        k = @(v1, v2) exp(-lambda * norm(v1 - v2, 1));  % Laplace kernel

        xx = [0:0.02:1];
        nn = length(xx);

        [X, Y] = meshgrid(xx);

        for p = [1:nn]
            for q = [1:nn]
                % Calculate prediction for every data point
                yi = 0;
                for i = [1:n]
                    yi = yi + alph(i) * y(i) * k(x(i, :), [X(p,q), Y(p,q)]);
                end
                yi = yi + b;
                ylist(p, q) = sign(yi);
            end
        end

        % Build heatmap
        subplot(4, 4, id);
        im = imagesc(xx, xx, ylist);
        im.AlphaData = .3;
        colormap autumn
        ax = gca;
        ax.YDir = 'normal';

        colors = [y == 1] * [1 0 0]; % red
        colors = colors + [y == -1] * [0 0 0]; % black
        hold on
        % Plot original values
        scatter(x(:, 1), x(:, 2), 6, colors);

        title(sprintf('lambda=%.1f, C=%.2f', lambda, C));

        id = id + 1;
        sprintf('Finished learning lambda=%f, C=%f', lambda, C)
    end
end

end
