import numpy
from numpy.core._exceptions import UFuncTypeError


class BaseOptimizer:
    """
    Base stochastic gradient descent optimizer.

    :param coef: array, initial coefficient
    :param learning_rate_init: float
        The initial learning rate used. It controls the step-size
        in updating the weights.
    :param min_threshold: coefficients must be higher than *min_thresold*
    :param max_threshold: coefficients must be below than *max_thresold*

    The class holds the following attributes:

    * *learning_rate*: float, the current learning rate
    * *coef*: optimized coefficients
    * *min_threshold*, *max_threshold*: coefficients thresholds
    * *l2*: L2 regularization
    * *l1*: L1 regularization
    """

    def __init__(
        self,
        coef,
        learning_rate_init=0.1,
        min_threshold=None,
        max_threshold=None,
        l1=0.0,
        l2=0.0,
    ):
        if not isinstance(coef, numpy.ndarray):
            raise TypeError("coef must be an array.")
        self.coef = coef
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)
        if min_threshold is not None and not isinstance(
            min_threshold, (float, numpy.float64, numpy.float32)
        ):
            raise TypeError("min_threshold must be a float")
        if max_threshold is not None and not isinstance(
            max_threshold, (float, numpy.float64, numpy.float32)
        ):
            raise TypeError("min_threshold must be a float")
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.l1 = l1
        self.l2 = l2

    def _get_updates(self, grad):
        raise NotImplementedError("Must be overwritten.")  # pragma no cover

    def update_coef(self, grad):
        """
        Updates coefficients with given gradient.

        :param grad: array, gradient
        """
        if self.coef.shape != grad.shape:
            raise ValueError(
                "coef and grad must have the same shape coef {} != gradient {}."
                "".format(self.coef.shape, grad.shape)
            )
        update = self._get_updates(grad)
        self.coef += update
        if self.min_threshold is not None:
            try:
                self.coef = numpy.maximum(self.coef, self.min_threshold)
            except UFuncTypeError:
                raise RuntimeError(  # noqa: B904
                    "Unable to compute an upper bound with coef={} "
                    "max_threshold={}".format(self.coef, self.min_threshold)
                )
        if self.max_threshold is not None:
            try:
                self.coef = numpy.minimum(self.coef, self.max_threshold)
            except UFuncTypeError:
                raise RuntimeError(  # noqa: B904
                    "Unable to compute a lower bound with coef={} "
                    "max_threshold={}".format(self.coef, self.max_threshold)
                )

    def iteration_ends(self, time_step):
        """
        Performs update to learning rate and potentially other states at the
        end of an iteration.
        """

    def train(
        self, X, y, fct_loss, fct_grad, max_iter=100, early_th=None, verbose=False
    ):
        """
        Optimizes the coefficients.

        :param X: datasets (array)
        :param y: expected target
        :param fct_loss: loss function, signature: `f(coef, X, y) -> float`
        :param fct_grad: gradient function,
            signature: `g(coef, x, y, i) -> array`
        :param max_iter: number maximum of iteration
        :param early_th: stops the training if the error goes below
            this threshold
        :param verbose: display information
        :return: loss

        The method keeps the best coefficients for the
        minimal loss.
        """
        if not isinstance(X, numpy.ndarray):
            raise TypeError("X must be an array.")
        if not isinstance(y, numpy.ndarray):
            raise TypeError("y must be an array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if any(numpy.isnan(X.ravel())):
            raise ValueError("X contains nan value.")
        if any(numpy.isnan(y.ravel())):
            raise ValueError("y contains nan value.")

        loss = fct_loss(self.coef, X, y)
        losses = [loss]
        best_coef = None
        best_loss = None
        if verbose:
            self._display_progress(0, max_iter, loss)
        n_samples = 0
        for it in range(max_iter):
            irows = numpy.random.choice(X.shape[0], X.shape[0])
            for irow in irows:
                grad = fct_grad(self.coef, X[irow, :], y[irow], irow)
                self._regularize_gradient(grad)
                if isinstance(verbose, int) and verbose >= 10:
                    self._display_progress(0, max_iter, loss, grad, "grad")
                if numpy.isnan(grad).sum() > 0:
                    raise RuntimeError("The gradient has nan values.")
                self.update_coef(grad)
                n_samples += 1

            self.iteration_ends(n_samples)
            loss = fct_loss(self.coef, X, y) + self.loss_regularization(self.coef)
            if verbose:
                self._display_progress(it + 1, max_iter, loss)
            self.iter_ = it + 1
            losses.append(loss)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_coef = self.coef.copy()
            if self._evaluate_early_stopping(
                it, max_iter, losses, early_th, verbose=verbose
            ):
                break
        self.coef = best_coef
        return best_loss

    def loss_regularization(self, coef):
        loss = 0
        if self.l1 > 0:
            loss += numpy.sum(numpy.abs(coef)) * self.l1
        if self.l2 > 0:
            loss += numpy.sum(coef**2) * self.l2
        return loss

    def _regularize_gradient(self, grad):
        """
        Applies regularization.
        """
        self.velocity_grad = grad
        if self.l2 > 0:
            grad += self.coef * self.l2
        if self.l1 > 0:
            grad += numpy.sign(self.coef) * self.l1

    def _evaluate_early_stopping(self, it, max_iter, losses, early_th, verbose=False):
        if len(losses) < 5 or early_th is None:
            return False
        if numpy.isnan(losses[-5]):
            if numpy.isnan(losses[-1]):
                if verbose:
                    self._display_progress(
                        it + 1, max_iter, losses[-1], losses=losses[-5:]
                    )
                return True
            return False
        if numpy.isnan(losses[-1]):
            if verbose:
                self._display_progress(it + 1, max_iter, losses[-1], losses=losses[-5:])
            return True
        if abs(losses[-1] - losses[-5]) <= early_th:
            if verbose:
                self._display_progress(it + 1, max_iter, losses[-1], losses=losses[-5:])
            return True
        return False

    def _display_progress(self, it, max_iter, loss, losses=None, msg=None):
        "Displays training progress."
        mxc = numpy.abs(self.coef.ravel()).max()
        l1 = numpy.sum(numpy.abs(self.coef))
        l2 = numpy.sum(self.coef * self.coef)
        vl1 = numpy.sum(numpy.abs(self.velocity_grad))
        vl2 = numpy.sum(self.velocity_grad * self.velocity_grad)
        if losses is None:
            print(
                "{}/{}: loss: {:1.4g} max(coef): {:1.2g} "
                "l1={:1.2g}/{:1.2g} l2={:1.2g}/{:1.2g}".format(
                    it, max_iter, loss, mxc, vl1, l1, vl2, l2
                )
            )
        else:
            print(
                "{}/{}: loss: {:1.4g} losses: {} max(coef): {:1.4g} "
                "l1={:1.2g}/{:1.2g} l2={:1.2g}/{:1.2g}".format(
                    it, max_iter, loss, losses, mxc, vl1, l1, vl2, l2
                )
            )


class SGDOptimizer(BaseOptimizer):
    """
    Stochastic gradient descent optimizer with momentum.

    :param coef: array, initial coefficient
    :param learning_rate_init: float
        The initial learning rate used. It controls the step-size
        in updating the weights,
    :param lr_schedule: `{'constant', 'adaptive', 'invscaling'}`,
        learning rate schedule for weight updates,
        `'constant'` for a constant learning rate given by
        *learning_rate_init*. `'invscaling'` gradually decreases
        the learning rate *learning_rate_* at each time step *t*
        using an inverse scaling exponent of *power_t*.
        `learning_rate_ = learning_rate_init / pow(t, power_t)`,
        `'adaptive'`, keeps the learning rate constant to
        *learning_rate_init* as long as the training keeps decreasing.
        Each time 2 consecutive epochs fail to decrease the training loss by
        tol, or fail to increase validation score by tol if 'early_stopping'
        is on, the current learning rate is divided by 5.
    :param momentum: float
        Value of momentum used, must be larger than or equal to 0
    :param power_t: double
        The exponent for inverse scaling learning rate.
    :param early_th: stops if the error goes below that threshold
    :param min_threshold: lower bound for parameters (can be None)
    :param max_threshold: upper bound for parameters (can be None)
    :param l1: L1 regularization
    :param l2: L2 regularization

    The class holds the following attributes:

    * *learning_rate*: float, the current learning rate
    * velocity*: array, velocity that are used to update params

    .. exref::
        :title: Stochastic Gradient Descent applied to linear regression

        The following example how to optimize a simple linear regression.

        .. runpython::
            :showcode:

            import numpy
            from mlstatpy.optim import SGDOptimizer


            def fct_loss(c, X, y):
                return numpy.linalg.norm(X @ c - y) ** 2


            def fct_grad(c, x, y, i=0):
                return x * (x @ c - y) * 0.1


            coef = numpy.array([0.5, 0.6, -0.7])
            X = numpy.random.randn(10, 3)
            y = X @ coef

            sgd = SGDOptimizer(numpy.random.randn(3))
            sgd.train(X, y, fct_loss, fct_grad, max_iter=15, verbose=True)
            print('optimized coefficients:', sgd.coef)
    """

    def __init__(
        self,
        coef,
        learning_rate_init=0.1,
        lr_schedule="invscaling",
        momentum=0.9,
        power_t=0.5,
        early_th=None,
        min_threshold=None,
        max_threshold=None,
        l1=0.0,
        l2=0.0,
    ):
        super().__init__(
            coef,
            learning_rate_init,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            l1=l1,
            l2=l2,
        )
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.power_t = power_t
        self.early_th = early_th
        self.velocity = numpy.zeros_like(coef)
        self.velocity_grad = numpy.zeros_like(coef)

    def iteration_ends(self, time_step):
        """
        Performs updates to learning rate and potential other states at the
        end of an iteration.

        :param time_step: int
            number of training samples trained on so far, used to update
            learning rate for 'invscaling'
        """
        if self.lr_schedule == "invscaling":
            self.learning_rate = (
                float(self.learning_rate_init) / (time_step + 1) ** self.power_t
            )
        elif self.lr_schedule == "constant":
            pass
        else:
            raise ValueError(f"Unexpected value: lr_schedule='{self.lr_schedule}'.")

    def _get_updates(self, grad):
        """
        Gets the values used to update params with given gradients.

        :param grad: array, gradient
        :return: updates, array, the values to add to params
        """
        update = self.momentum * self.velocity - self.learning_rate * grad
        self.velocity = update
        return update

    def _display_progress(self, it, max_iter, loss, losses=None, msg="loss"):
        "Displays training progress."
        mxc = numpy.abs(self.coef.ravel()).max()
        l1 = numpy.sum(numpy.abs(self.coef))
        l2 = numpy.sum(self.coef * self.coef)
        vl1 = numpy.sum(numpy.abs(self.velocity_grad))
        vl2 = numpy.sum(self.velocity_grad * self.velocity_grad)
        if losses is None:
            print(
                "{}/{}: {}: {:1.4g} lr={:1.3g} max(coef): {:1.2g} "
                "l1={:1.2g}/{:1.2g} l2={:1.2g}/{:1.2g}".format(
                    it, max_iter, msg, loss, self.learning_rate, mxc, vl1, l1, vl2, l2
                )
            )
        else:
            print(
                "{}/{}: {}: {:1.4g} lr={:1.3g} {}es: {} "
                "max(coef): {:1.2g} l1={:1.2g}/{:1.2g} "
                "l2={:1.2g}/{:1.2g}".format(
                    it,
                    max_iter,
                    msg,
                    loss,
                    self.learning_rate,
                    msg,
                    losses,
                    mxc,
                    vl1,
                    l1,
                    vl2,
                    l2,
                )
            )
