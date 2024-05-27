import torch as ch
from torch.distributions import Categorical, MultivariateNormal
from toy_task.GMM.models.model_factory import get_model
from toy_task.GMM.targets.gaussian_mixture_target import  get_gmm_target, get_weights
from toy_task.GMM.algorithms.evaluation.JensenShannon_Div import js_divergence


class SimpleGMM:
    def __init__(self, means, chols, weights, device):
        self.means = means
        self.chols = chols
        self.weights = weights
        self.device = device

    def get_samples_gmm(self, log_gates, means, chols, num_samples):
        if log_gates.shape[1] == 1:
            # print("wrong branch")
            samples = MultivariateNormal(means.squeeze(1), scale_tril=chols.squeeze(1)).sample((num_samples,))
            return samples.transpose(0, 1)
        else:
            samples = []
            for i in range(log_gates.shape[0]):
                cat = Categorical(log_gates[i])
                indices = cat.sample((num_samples,))
                chosen_means = means[i, indices]
                chosen_chols = chols[i, indices]
                normal = MultivariateNormal(chosen_means, scale_tril=chosen_chols)
                samples.append(normal.sample())
            return ch.stack(samples)  # [n_contexts, n_samples, n_features

    def log_prob_gmm(self, means, chols, log_gates, samples):
        n_samples = samples.shape[1]
        n_contexts, n_components, _ = means.shape

        means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
        chols_expanded = chols.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)

        mvn = MultivariateNormal(means_expanded, scale_tril=chols_expanded)
        log_probs = mvn.log_prob(samples_expanded)  # [batch_size, n_samples, n_components]

        gate_expanded = log_gates.unsqueeze(1).expand(-1, n_samples, -1)
        log_probs += gate_expanded

        log_probs = ch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
        # return ch.sum(log_probs, dim=1)
        return log_probs

    def __call__(self, contexts):
        gate = self.weights
        mean = self.means
        chol = self.chols
        return gate, mean, chol


device = ch.device('cuda' if ch.cuda.is_available() else 'cpu')
n_contexts = 200
target = get_gmm_target(4)
eval_contexts = target.get_contexts(n_contexts).to(device)
model_type = "real"  # "real", "random" or "target"

if model_type == "target":
    # target distribution same as the target
    batched_means = target.mean_fn(eval_contexts).to(device)
    batched_chols = target.chol_fn(eval_contexts).to(device)
    batched_weights = get_weights(eval_contexts).to(device)
    model = SimpleGMM(batched_means, batched_chols, batched_weights, device=device)

elif model_type == "random":
    # random target distribution, with 4 components
    means = ch.tensor([[0, 0], [1., 1.], [2., 3.], [4., 5.]])
    chols = ch.tensor([[[1., 0], [0, 1.]], [[5., 0], [0, 1.]], [[1., 0], [0, 3.]], [[3., 0], [0, 2.]]])
    weights = ch.tensor([0.1, 0.3, 0.2, 0.4])
    batched_means = means.unsqueeze(0).expand(n_contexts, -1, -1).to(device)
    batched_chols = chols.unsqueeze(0).expand(n_contexts, -1, -1, -1).to(device)
    batched_weights = weights.unsqueeze(0).expand(n_contexts, -1).to(device)
    model = SimpleGMM(batched_means, batched_chols, batched_weights, device=device)

elif model_type == "real":
    # real target but no training
    model = get_model(model_name="toy_task_model_1",
                      device=device,
                      dim=2,
                      fc_layer_size=256,
                      n_components=4,
                      initialization_type="xavier")

else:
    raise ValueError("model_type should be one of 'real', 'random' or 'target'")


# Calculate JSD
jsd_value = js_divergence(model, target, eval_contexts, device)
print("Calculated JSD:", jsd_value.item())
