clear;
% open csv file
tbl = readtable('test.csv');

% replace strings fields with labels
ds(:,1) = grp2idx(tbl{:,2});
ds(:,2) = grp2idx(tbl{:,3});
ds(:,3) = tbl{:,4};
ds(:,4) = grp2idx(tbl{:,5});
ds(:,5) = grp2idx(tbl{:,6});
ds(:,6) = tbl{:,7};
ds(:,7) = grp2idx(tbl{:,8});
ds(:,8) = tbl{:,9};
[ds(:,9),labels] = grp2idx(tbl{:,10});

% remove rows with NaN in any field values
ds = rmmissing(ds); 
[m,n] = size(ds);

X = ds(:,[2 4 8]); 
y = ds(:,n);

input_layer_size  = size(X,2);              % Dimension of features 
hidden_layer_size = input_layer_size*2;     % number of units in hidden layer (twice input layer a good start)
output_layer_size = size(labels,1);         % number of labels

% randominitialize 2 thetas for hidden layer and output layer
initial_Theta1 = initializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = initializeWeights(hidden_layer_size, output_layer_size);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];  % Unroll parameters

% Run cost function to find lowest cost thetas
options = optimset('MaxIter', 50);
lambda = 1;
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, output_layer_size, X, y, lambda);
[nn_params, ~] = fminunc(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), output_layer_size, (hidden_layer_size + 1));
 
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);