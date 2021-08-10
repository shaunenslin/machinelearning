from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('Life Expectancy Data.csv')
df.head()

df.rename(columns={" BMI ": "BMI",
                   "Life expectancy ": "Life_expectancy",
                   "Adult Mortality": "Adult_mortality",
                   "infant deaths": "Infant_deaths",
                   "percentage expenditure": "Percentage_expenditure",
                   "Hepatitis B": "HepatitisB",
                   "Measles ": "Measles",
                   "under-five deaths ": "Under_five_deaths",
                   "Total expenditure": "Total_expenditure",
                   "Diphtheria ": "Diphtheria",
                   " thinness  1-19 years": "Thinness_1-19_years",
                   " thinness 5-9 years": "Thinness_5-9_years",
                   " HIV/AIDS": "HIV/AIDS",
                   "Income composition of resources": "Income_composition_of_resources"}, inplace=True)

columns = ["Status", "Country"]
for feature in columns:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])

Y = df["Life_expectancy"]
df = df.drop(["Life_expectancy"], axis=1)


def fillmissing(df, feature, method):
    if method == "mode":
        df[feature] = df[feature].fillna(df[feature].mode()[0])

    elif method == "median":
        df[feature] = df[feature].fillna(df[feature].median())

    else:
        df[feature] = df[feature].fillna(df[feature].mean())


features_missing = df.columns[df.isna().any()]
for feature in features_missing:
    fillmissing(df, feature=feature, method="mean")

X = df.to_numpy()  # np.matrix(df.to_numpy())
Y.fillna(Y.median(), inplace=True)
y = Y.to_numpy().transpose()  # np.matrix(Y.to_numpy()).transpose()

m, n = X.shape

mu = X.mean()
sigma = X.std()  # standard deviation: max(x)-min(x)
xn = (X - mu) / sigma
#xn2 = (X - X.mean()) / (X.max() - X.min())


# np.matrix(np.hstack((np.ones((m, 1)), xn)))
xo = np.hstack((np.ones((m, 1)), xn))

repeat = 100
lrate = 0.5
theta2 = np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0]).transpose()
theta = np.zeros((n+1))
costhistory = np.zeros((repeat, 1))


def computeCostMulti(X, y, theta):
    """Compute cost for linear regression with multiple variables

    J = computeCostMulti(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    :param X:
    :param y:
    :param theta:
    :return:
    """
    # Initialize some useful values
    m = len(y)  # number of training examples
    diff = np.matmul(X, theta) - y
    J = 1 / (2 * m) * np.matmul(diff, diff)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta

    theta = gradientDescentMulti(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    :param X:
    :param y:
    :param theta:
    :param alpha:
    :param num_iters:
    :return:
    """
    # Initialize some useful values
    m = len(y)  # number of training examples
    J_history = []

    for i in range(num_iters):
        theta -= alpha / m * np.matmul(X.transpose(), np.matmul(X, theta) - y)
        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


theta, J_history = gradientDescent(xo, y, theta, lrate, repeat)

# Plot the convergence graph
plt.plot(np.arange(repeat), J_history, '-b', LineWidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(f' {theta} ')

# #costhistory = pd.DataFrame([0] * repetition, columns=['cost'])
# for i in range(1):
#     # calculate cost of hypothesis
#     hc = xo * theta - y
#     temp = hc.transpose() * xo
#     # new theta
#     theta = theta - (lrate * (1/m)) * temp.transpose()
#     # recalculate cost of hypothesis with new theta
#     hc = xo * theta - y
#     costhistory[i] = (hc.transpose() * hc) / (2*m)
