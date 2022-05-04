import numpy as np
import torch
import torch.optim as optim
from typing import Optional, Union

BIG_STD = 20.  # used as std of prior when it should be uninformative (when we do not wish to regularize at all)


def to_tensor(x):
    return torch.from_numpy(np.array(x)).float()


class BatchElasticNetRegression(object):
    """
    Elastic net for the case where we have multiple targets (y), all to be fitted with the same features (X).
    Learning all items in parallel, in a single "network" is more efficient then iteratively fitting a regression for
    each target.
    Allows to set different l1 and l2 regularization params for each of the features.
    Can also be used to estimate the MAP of a Bayesian regression with Laplace or Normal priors instead of L1 and L2.
    """
    def __init__(self):
        self.coefs = None
        self.intercepts = None

    def fit(self,
            X, y,
            l1_reg_params: Optional[Union[np.array, float]] = None,
            l2_reg_params: Optional[Union[np.array, float]] = None,
            as_bayesian_prior=False, iterations=500, verbose=True, lr_rate=0.1):
        """
        Fits multiple regressions. Both X and y are 2d matrices, where X is the common features for all the targets,
        and y contains all the concatenated targets.
        If as_bayesian_prior==False then the l1 and l2 reg params are regularization params
        If as_bayesian_prior==True then l1 is treated as the std of the laplace prior and l2 as the std for the normal
        prior.
        The reg params / std of priors can either be a single value for all features, or set a different regularization
        or prior for each feature separately. e.g. if we have 3 features, l1_reg_params can be [0.5, 1.2, 0] to set
        regularization for each.

        TODO:
        Add normalization before fitting
        Requires more work on the optimizer to be faster
        """
        n_items = y.shape[1]
        k_features = X.shape[1]
        n_samples = X.shape[0]

        # TODO: if l1_reg_params is None just don't calculate this part of the loss, instead of multiplying by 0
        if l1_reg_params is None:
            l1_reg_params = BIG_STD if as_bayesian_prior else 0.
        if type(l1_reg_params) == float:
            l1_reg_params = [l1_reg_params] * k_features
        if l2_reg_params is None:
            l2_reg_params = BIG_STD if as_bayesian_prior else 0.
        if type(l2_reg_params) == float:
            l2_reg_params = [l2_reg_params] * k_features

        assert len(l1_reg_params) == len(l2_reg_params) == k_features, 'Regularization values must match X.shape[1]'
        if as_bayesian_prior:
            assert 0 not in l1_reg_params and 0 not in l2_reg_params, 'Cannot have 0 prior'

        # convert to tensors and set initial params
        t_features = to_tensor(X)
        t_target = to_tensor(y)
        learned_coefs = torch.rand(k_features, n_items, requires_grad=True)
        learned_intercepts = torch.rand(n_items, requires_grad=True)
        # TODO: or auto-estimate initial sigma based on data std?
        est_sigma = torch.ones(n_items)
        if as_bayesian_prior:
            # If the params are priors then they must become a matrix, not a simple vector - because the conversion
            # depends on the sigma of errors for each target y. The actual regularization params will be different
            # for each item based on its sigma.
            t_l1_reg_params = to_tensor(np.stack([l1_reg_params] * n_items, axis=1))
            l1_alpha = self.calc_l1_alpha_from_prior(est_sigma, t_l1_reg_params, n_samples)
            t_l2_reg_params = to_tensor(np.stack([l2_reg_params] * n_items, axis=1))
            l2_alpha = self.calc_l2_alpha_from_prior(est_sigma, t_l2_reg_params, n_samples)
        else:
            l1_alpha = to_tensor(l1_reg_params)
            l2_alpha = to_tensor(l2_reg_params)

        # TODO: add scheduler for learning rate
        optimizer = optim.Adam([learned_coefs, learned_intercepts], lr_rate)

        for i in range(iterations):
            optimizer.zero_grad()
            res = torch.matmul(t_features, learned_coefs) + learned_intercepts
            diff_loss = (1 / (2 * n_samples)) * ((res - t_target) ** 2).sum(axis=0)

            if as_bayesian_prior:
                reg_loss = (l1_alpha * learned_coefs.abs()).sum(axis=0) + (l2_alpha * learned_coefs ** 2).sum(axis=0)
            else:
                reg_loss = torch.matmul(l1_alpha, learned_coefs.abs()) + torch.matmul(l2_alpha, learned_coefs ** 2)

            loss = (diff_loss + reg_loss).sum()

            loss.backward()
            optimizer.step()
            if as_bayesian_prior and i % 50 == 0:
                # if the params are the priors - we must convert them to the equivalent l1/l2 loss params.
                # This conversion depends on the final sigma of errors of the forecast, which is unknown until we
                # have a forecast using those same params... We iteratively improve our estimate of sigma and
                # re-compute the corresponding regularization params based on those sigmas.
                # The sigma is per target in y, therefore the l1/l2 params are per item.
                est_sigma = (res - t_target).std(axis=0).detach()
                l1_alpha = self.calc_l1_alpha_from_prior(est_sigma, t_l1_reg_params, n_samples)
                l2_alpha = self.calc_l2_alpha_from_prior(est_sigma, t_l2_reg_params, n_samples)

            if i % 50 == 0 and verbose:
                print(loss)
            # TODO: early stopping if converges

        self.coefs = learned_coefs.detach().numpy()
        self.intercepts = learned_intercepts.detach().numpy()

    def predict(self, X):
        return X @ self.coefs + self.intercepts

    @staticmethod
    def calc_l1_alpha_from_prior(est_sigma, b_prior, n_samples):
        """
        Converts from the std of a Laplace prior to the equivalent L1 regularization param.
        The conversion formula is divided by 2*n_samples since we divided the diff_loss by 2*n_samples as well,
        to match sklearn's implementation of Lasso.
        """
        return est_sigma ** 2 / (b_prior * n_samples)

    @staticmethod
    def calc_l2_alpha_from_prior(est_sigma, b_prior, n_samples):
        return est_sigma ** 2 / (b_prior ** 2 * 2 * n_samples)
