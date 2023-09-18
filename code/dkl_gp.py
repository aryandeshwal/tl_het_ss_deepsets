import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from botorch.posteriors.gpytorch import GPyTorchPosterior
from deepsets_embedding import DeepEmbeddingHetSpaces
import torch


class GPModelDKL(ApproximateGP):
    def __init__(
        self,
        inducing_points_set,
        likelihood,
        network_dims=(32, 32),
        learn_inducing_locations=False,
    ):
        feature_extractor = DeepEmbeddingHetSpaces(network_dims=network_dims).cuda()
        inducing_points = []
        for ind_point in inducing_points_set:
            with torch.no_grad():
                inducing_points.append(feature_extractor(*ind_point[:3]))
        inducing_points = torch.stack(inducing_points)
        inducing_points = inducing_points.view(-1, inducing_points.shape[-1])
        print(inducing_points.shape)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super(GPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1  # must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, sup_x, sup_y, que_x, *args, **kwargs):
        x = self.feature_extractor(
            sup_x,
            sup_y,
            que_x,
        )
        return super().__call__(x, *args, **kwargs)

    def posterior(self, sup_x, sup_y, que_x, *args, **kwargs) -> GPyTorchPosterior:
        self.eval()
        self.likelihood.eval()
        dist = self.likelihood(
            self(
                sup_x,
                sup_y,
                que_x,
            )
        )
        return GPyTorchPosterior(mvn=dist)
