import torch as ch

from toy_task.GMM.models.model_factory import get_model


# test abstract model methods
n_contexts = 10
n_samples = 5
n_components = 3
dim = 2

gate = ch.rand(n_contexts, n_components)
log_gates = gate / gate.sum(dim=-1, keepdim=True)
means = ch.randn(n_contexts, n_components, dim)
covs = ch.eye(dim).repeat(n_contexts, n_components, 1, 1)
chols = ch.linalg.cholesky(covs)

model = get_model(model_name="toy_task_model_3",
                  target_name="gmm",
                  dim=dim,
                  context_dim=2,
                  random_init=True,
                  device="cpu",
                  max_components=5,
                  init_components=5,
                  gate_layer=3,
                  com_layer=5,
                  initialization_type="xavier")

samples = model.get_rsamples(means, chols, n_samples)
log_probs = model.log_prob(means, chols, samples)
assert samples.shape == (n_samples, n_contexts, n_components, dim)
assert log_probs.shape == (n_samples, n_contexts, n_components)

samples_gmm = model.get_samples_gmm(log_gates, means, chols, n_samples)
log_probs_gmm = model.log_prob_gmm(means, chols, log_gates, samples_gmm)
assert samples_gmm.shape == (n_samples, n_contexts, dim)
assert log_probs_gmm.shape == (n_samples, n_contexts)

log_responsibilities = model.log_responsibilities_gmm(means, chols, log_gates, samples)
assert log_responsibilities.shape == (n_samples, n_contexts, n_components)

