import torch as ch


from toy_task.GMM.utils.network_utils import set_seed


if __name__ == "__main__":

    set_seed(1001)

    def test_gmm_target(n_components=10, context_dim=2, n_contexts=3, dz=2, n_samples=5):
        from toy_task.GMM.targets.target_factory import get_gmm_target

        target = get_gmm_target(n_components, context_dim)
        test_ctx = target.get_contexts(n_contexts)  # (3, 2)
        assert test_ctx.shape == (n_contexts, context_dim)

        test_ctx = test_ctx.cuda()
        test_samples = target.sample(test_ctx, n_samples)
        assert test_samples.shape == (n_samples, n_contexts, dz)

        test_samples.requires_grad = True
        test_log_prob = target.log_prob_tgt(test_ctx, test_samples)
        assert test_log_prob.shape == (n_samples, n_contexts)
        # target.visualize(test_ctx, n_samples=20)
        # target.visualize(test_ctx)

        samples = ch.randn(n_samples, n_contexts, dz)
        updated_samples = target.update_samples((test_ctx, samples), target.log_prob_tgt, 0.1, 100)
        print(samples)
        print(updated_samples)


    def test_star_target(n_components=7, context_dim=2, n_contexts=3, dz=2, n_samples=5):
        from toy_task.GMM.targets.target_factory import get_star_target

        target = get_star_target(n_components, context_dim)
        contexts = target.get_contexts(n_contexts)
        # samples = target.sample(contexts, n_samples)
        # target.visualize(contexts, n_samples=20)
        target.visualize(contexts)


    def test_funnel_target(context_dim=2, n_contexts=3, n_samples=5):
        from toy_task.GMM.targets.funnel_target import FunnelTarget, get_sig_fn

        target = FunnelTarget(get_sig_fn, context_dim)
        contexts = target.get_contexts(n_contexts)
        samples = target.sample(contexts, n_samples)
        target.visualize(contexts, n_samples=20)
        target.visualize(contexts)

    test_funnel_target()
