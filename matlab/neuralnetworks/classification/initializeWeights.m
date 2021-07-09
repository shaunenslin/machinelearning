function W = initializeWeights(L_in, L_out)
    % Randomly initialize the weights to small values
    epsilon_init = 0.12;
    W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end
