import numpy as np

from linreg import LinearRegression

if __name__ == "__main__":
    '''
        Main function to test multivariate linear regression
    '''

    # load the data
    import pickle

    with open('holdout.pickle', 'rb') as f:
        testdata = pickle.load(f)
    # filePath = 'multivariateData.dat'
    # file = open(filePath, 'r')
    # allData = np.loadtxt(file, delimiter=',')

    X_test = np.matrix(testdata[:, :-1])
    y_test = np.matrix((testdata[:, -1])).T

    n, d = X_test.shape

    # Standardize
    mean = X_test.mean(axis=0)
    std = X_test.std(axis=0)
    X_test = (X_test - mean) / std

    # Add a row of ones for the bias term
    X_test = np.c_[np.ones((n, 1)), X_test]

    # initialize the model
    init_theta = np.matrix(np.random.randn((d + 1))).T
    n_iter = 2000
    alpha = 0.01

    # Instantiate objects
    lr_model = LinearRegression(init_theta=init_theta, alpha=alpha, n_iter=n_iter)
    lr_model.fit(X_test, y_test)
