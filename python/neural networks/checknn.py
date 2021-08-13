import numpy as np
from matplotlib import pyplot as plt


def sigmoid(z):
    g = np.frompyfunc(lambda x: 1 / (1 + np.exp(-x)), 1, 1)
    return g(z).astype(z.dtype)


def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def nnCostFunction2(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    """Implements the neural network cost function for a two layer
    neural network which performs classification

    [J grad] = nnCostFunction(nn_params, hidden_layer_size, num_labels, ...
    X, y, lambda) computes the cost and gradient of the neural network.
    :param nn_params: "unrolled" parameters for the neural network,
    need to be converted back into the weight matrices.
    :param input_layer_size:
    :param hidden_layer_size:
    :param num_labels:
    :param X:
    :param y:
    :param lambda_:
    :return: grad should be a "unrolled" vector of the
    partial derivatives of the neural network.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
        (hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size *
                       (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    # Setup some useful variables
    m = X.shape[0]

    # Add ones to the X data matrix
    X = np.insert(X, 0, 1, axis=1)

    z2 = np.matmul(X, Theta1.transpose())
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = np.matmul(a2, Theta2.transpose())
    a3 = sigmoid(z3)

    y_one_hot = np.zeros_like(a3)
    for i in range(m):
        y_one_hot[i, y[i] - 1] = 1

    ones = np.ones_like(a3)
    A = np.matmul(y_one_hot.transpose(), np.log(a3)) + \
        np.matmul((ones - y_one_hot).transpose(), np.log(ones - a3))
    J = -1 / m * A.trace()
    J += lambda_ / (2 * m) * \
        (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    delta3 = a3 - y_one_hot
    delta2 = np.matmul(delta3, Theta2[:, 1:]) * sigmoidGradient(z2)
    Theta2_grad = np.matmul(a2.transpose(), delta3).transpose()
    Theta1_grad = np.matmul(X.transpose(), delta2).transpose()

    Theta1_grad[:, 1:] += lambda_ * Theta1[:, 1:]
    Theta2_grad[:, 1:] += lambda_ * Theta2[:, 1:]
    Theta1_grad /= m
    Theta2_grad /= m
    grad = np.concatenate([Theta1_grad.reshape(-1), Theta2_grad.reshape(-1)])
    return J, grad


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
        (hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size *
                       (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    # Setup some useful variables
    m = X.shape[0]

    # Add ones to the X data matrix
    a1 = np.insert(X, 0, 1, axis=1)

    # Perform forward propagation for layer 2
    z2 = np.matmul(a1, Theta1.transpose())
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)

    # perform forward propagation for layer 3
    z3 = np.matmul(a2, Theta2.transpose())
    a3 = sigmoid(z3)

    # turn Y into a matrix with a new column for each category and marked with 1
    yv = np.zeros_like(a3)
    for i in range(m):
        yv[i, y[i] - 1] = 1

    # calculate penalty without theta0
    p = sum(sum(np.power(Theta1[:, 1:], 2))) + \
        sum(sum(np.power(Theta2[:, 1:], 2)))

    # Calculate the cost of our forward prop
    J = sum(sum(-yv * np.log(a3) - (1 - yv) * np.log(1 - a3), 2)) / \
        (m + lambda_ * p/(2*m))

    # Perform backward propagation to calculate deltas
    s3 = a3 - yv
    # s2 = (s3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]); % remove z2 bias column
    s2 = np.matmul(s3, Theta2) * \
        sigmoidGradient(np.insert(z2, 0, 1, axis=1))
    s2 = s2[:, 1:]
    #s2 = np.matmul(s3, Theta2[:, 1:]) * sigmoidGradient(z2)

    # Calculate DELTA's (accumulated deltas)
    delta_1 = np.matmul(s2.transpose(), a1)
    delta_2 = np.matmul(s3.transpose(), a2)

    # calculate regularized gradient, replace 1st column with zeros
    p1 = (lambda_/m) * np.insert(Theta1[:, 1:], 0, 0, axis=1)
    p2 = (lambda_/m) * np.insert(Theta2[:, 1:], 0, 0, axis=1)

    # gradients / partial derivitives
    Theta1_grad = delta_1 / m + p1
    Theta2_grad = delta_2 / m + p2
    grad = np.concatenate(
        (Theta1_grad.flatten(), Theta2_grad.flatten()), axis=None)

    # # unroll gradients
    return J, grad


def checkNNGradients(lambda_=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.arange(1, m + 1) % num_labels

    # Unroll parameters
    nn_params = np.concatenate([Theta1.reshape(-1), Theta2.reshape(-1)])

    # Short hand for cost function
    def costFunc(p): return nnCostFunction(p, input_layer_size,
                                           hidden_layer_size, num_labels, X, y, lambda_)

    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.The two columns
    # you get should be very similar.
    print(np.column_stack([numgrad, grad]))
    print('The above two columns you get should be very similar.\n'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          f'\nRelative Difference : {diff:g}')


def debugInitializeWeights(fan_out, fan_in):
    """Initialize the weights of a layer with fan_in
    incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging

    W = debugInitializeWeights(fan_in, fan_out) initializes the weights
    of a layer with fan_in incoming connections and fan_out outgoing
    connections using a fix set of values
    :param fan_out:
    :param fan_in:
    :return: W: a matrix of size(1 + fan_in, fan_out) as
    the first row of W handles the "bias" terms
    """
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.sin(np.arange(1, fan_out * (1 + fan_in) + 1)
               ).reshape((fan_out, 1 + fan_in)) / 10
    return W


def computeNumericalGradient(J, theta):
    """Computes the gradient using "finite differences"
    and gives us a numerical estimate of the gradient.

    numgrad = computeNumericalGradient(J, theta) computes the numerical
    gradient of the function J around theta. Calling y = J(theta) should
    return the function value at theta.
    :param J:
    :param theta:
    :return: numgrad(i): a numerical approximation of)
    the partial derivative of J with respect to the
    i-th input argument, evaluated at theta.
    """
    numgrad = np.zeros_like(theta).reshape(-1)
    perturb = np.zeros_like(theta).reshape(-1)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        loss1, _ = J(theta - perturb.reshape(theta.shape))
        loss2, _ = J(theta + perturb.reshape(theta.shape))
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad.reshape(theta.shape)


def predict(Theta1, Theta2, X):
    """Predict the label of an input given a trained neural network

    p = predict(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    :param Theta1:
    :param Theta2:
    :param X:
    :return:
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # Add ones to the X data matrix
    X = np.insert(X, 0, 1, axis=1)

    a2 = sigmoid(np.matmul(X, Theta1.transpose()))
    a2 = np.insert(a2, 0, 1, axis=1)
    a3 = sigmoid(np.matmul(a2, Theta2.transpose()))
    p = a3.argmax(axis=1) + 1
    return p
