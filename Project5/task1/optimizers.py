import numpy as np
from cmath import inf


class Regressor:

    def __init__(self) -> None:
        self.X, self.y = self.generate_dataset(n_samples=200, n_features=1)
        self.n, d = self.X.shape
        self.w = np.zeros((d, 1))

    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """
        from sklearn.datasets import make_regression

        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=30)
        y = y.reshape(n_samples, 1)
        return X, y

    def linear_regression(self, X):
        """
        Performs linear regression on a dataset
        Returns:
            y: a numpy.ndarray of shape (n, 1) containing the predictions
        """
        y = np.dot(X, self.w)
        return y

    def predict(self, X):
        """
        Predicts the labels for a given dataset
        X: a numpy.ndarray of shape (n, d) containing the dataset
        Returns:
            y: a numpy.ndarray of shape (n,) containing the predictions
        """
        y = np.dot(X, self.w).reshape(X.shape[0])
        return y

    def compute_loss(self, X, y):
        """
        Computes the MSE loss of a prediction
        Returns:
            loss: the loss of the prediction
        """
        predictions = self.linear_regression(X)
        loss = np.mean((predictions - self.y) ** 2)
        return loss

    def compute_gradient(self, X, y):
        """
        Computes the gradient of the MSE loss
        Returns:
            grad: the gradient of the loss with respect to w
        """
        predictions = self.linear_regression(X)
        dif = (predictions - y)
        grad = 2 * np.dot(X.T, dif)
        return grad

    def fit(self, optimizer='adam', n_iters=1000, render_animation=True):
        """
        Trains the model
        optimizer: the optimization algorithm to use
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        n_iters: the number of iterations to train for
        """

        figs = []
        dw = 0
        m = 0
        v = 0
        best_loss = inf

        for i in range(1, n_iters + 1):

            if optimizer == 'gd':
                self.w = self.gradient_descent(alpha=0.003)
                pass
            elif optimizer == "sgd":
                self.w = self.sgd_optimizer(alpha=0.003, batch_size=20)
                pass
            elif optimizer == "sgdMomentum":
                self.w = self.sgd_momentum(alpha=0.003, batch_size=20, momentum=0.2)
                pass
            elif optimizer == "adagrad":
                self.w, v, dw = self.adagrad_optimizer(v, 80, dw, 0.01)
                pass
            elif optimizer == "rmsprop":
                self.w, v, dw = self.rmsprop_optimizer(v, 50, 0.1, dw, 0.01)
                pass
            elif optimizer == "adam":
                self.w, v, m, dw = self.adam_optimizer(m, v, 5, 0.02, 0.01, dw, i, 0.01)
                pass

            # TODO: implement the stop criterion

            if self.compute_loss(self.X, self.y) >= best_loss:
                break
            else:
                best_loss = self.compute_loss(self.X, self.y)


            #
            print("Iteration: ", i)
            print("Loss: ", self.compute_loss(self.X, self.y))

            if render_animation:
                import matplotlib.pyplot as plt
                from moviepy.video.io.bindings import mplfig_to_npimage

                fig = plt.figure()
                plt.scatter(self.X, self.y, color='red')
                plt.plot(self.X, self.predict(self.X), color='blue')
                plt.xlim(self.X.min(), self.X.max())
                plt.ylim(self.y.min(), self.y.max())
                plt.title(f'Optimizer:{optimizer}\nIteration: {i}')
                plt.close()
                figs.append(mplfig_to_npimage(fig))

        if render_animation and len(figs) > 0:
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(figs, fps=5)
            clip.write_gif(f'{optimizer}_animation.gif', fps=5)

    def gradient_descent(self, alpha):
        """
        Performs gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        # TODO: implement gradient descent

        diff = -alpha * (self.compute_gradient(self.X, self.y))
        self.w += diff
        return self.w

    def sgd_optimizer(self, alpha, batch_size):
        """
        Performs stochastic gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        x_y = np.c_[self.X.reshape(self.n, -1), self.y.reshape(self.n, 1)]
        rng = np.random.default_rng(seed=42)
        rng.shuffle(x_y)

        for start in range(0, self.n, batch_size):
            stop = start + batch_size
            x_batch, y_batch = x_y[start:stop, :-1], x_y[start:stop, -1:]

            """implement gd on batches"""

            gradient = self.compute_gradient(x_batch, y_batch)
            diff = - alpha * gradient

            self.w += diff

        return self.w

    def sgd_momentum(self, alpha, batch_size, momentum):
        """
        Performs SGD with momentum to optimize the weights
        alpha: the learning rate
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        # TODO: implement stochastic gradient descent
        x_y = np.c_[self.X.reshape(self.n, -1), self.y.reshape(self.n, 1)]
        rng = np.random.default_rng(seed=42)
        rng.shuffle(x_y)

        diff = 0

        for start in range(0, self.n, batch_size):
            stop = start + batch_size
            x_batch, y_batch = x_y[start:stop, :-1], x_y[start:stop, -1:]

            """implement gd on batches"""

            gradient = self.compute_gradient(x_batch, y_batch)
            diff = - alpha * gradient + momentum * diff

            self.w += diff

        return self.w

    def adagrad_optimizer(self, v, alpha, dw, epsilon):
        """
        Performs Adagrad optimization to optimize the weights
        alpha: the learning rate
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = self.w

        gradient = self.compute_gradient(self.X, self.y)
        dw += gradient
        v += dw ** 2

        w += -(alpha / np.sqrt(v + epsilon)) * gradient

        return self.w, v, dw

    def rmsprop_optimizer(self, v, alpha, beta, dw, epsilon):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients
        alpha: the learning rate
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = self.w

        gradient = self.compute_gradient(self.X, self.y)
        dw += gradient
        v = beta * v + (1 - beta) * dw ** 2

        w += -(alpha / np.sqrt(v + epsilon)) * gradient

        return self.w, v, dw

    def adam_optimizer(self, m, v, alpha, beta1, beta2, dw, i, epsilon):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        alpha: the learning rate
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        # TODO: implement stochastic gradient descent
        w = self.w

        gradient = self.compute_gradient(self.X, self.y)
        dw += gradient

        m = beta1 * m + (1 - beta1) * dw
        v = beta2 * v + (1 - beta2) * dw ** 2

        m = m / (1 - beta1 ** (i + 1))
        v = v / (1 - beta2 ** (i + 1))

        w += -(alpha / np.sqrt(v + epsilon)) * m
        return self.w, v, m, dw


r = Regressor()
r.fit()