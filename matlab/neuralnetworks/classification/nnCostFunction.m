function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
    m = size(X, 1);

    % perform forward propagation for layer 2
    a1 = [ones(m, 1) X];
    z2 = a1*Theta1';
    a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
    % perform forward propagation for layer 3
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    yv = [1:num_labels] == y;

    % calculate penalty without theta0, 
    p = sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:, 2:end).^2, 2));
    % Calculate the cost of our forward prop
    J = sum(sum(-yv .* log(a3) - (1 - yv) .* log(1-a3), 2))/m + lambda*p/(2*m);

    % perform backward propagation to calculate deltas
    s3 = a3 - yv;
    s2 = (s3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]); % remove z2 bias column
    s2 = s2(:, 2:end);

    % Calculate DELTA's (accumulated deltas)
    delta_1 = (s2'*a1);
    delta_2 = (s3'*a2);

    % calculate regularized gradient, replace 1st column with zeros
    p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
    p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
    
    % gradients / partial derivitives
    Theta1_grad = delta_1./m + p1;
    Theta2_grad = delta_2./m + p2;

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
