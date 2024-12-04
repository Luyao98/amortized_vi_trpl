import torch as ch


from toy_task.GMM.utils.network_utils import set_seed


if __name__ == "__main__":

    set_seed(1002)

    def test_gmm_target(n_components=10, context_dim=2, n_contexts=3, dz=2, n_samples=5):
        from toy_task.GMM.targets.target_factory import get_gmm_target

        target = get_gmm_target(n_components, context_dim)
        test_ctx = target.get_contexts(n_contexts)  # (3, 2)
        assert test_ctx.shape == (n_contexts, context_dim)

        # test_ctx = test_ctx.cuda()
        # test_samples = target.sample(test_ctx, n_samples)
        # assert test_samples.shape == (n_samples, n_contexts, dz)
        #
        # test_samples.requires_grad = True
        # test_log_prob = target.log_prob_tgt(test_ctx, test_samples)
        # assert test_log_prob.shape == (n_samples, n_contexts)
        # target.visualize(test_ctx, n_samples=20)
        target.visualize(test_ctx, filename="general_10_gmm_target.pdf")

        # samples = ch.randn(n_samples, n_contexts, dz)
        # updated_samples = target.update_samples((test_ctx, samples), target.log_prob_tgt, 0.1, 100)
        # print(samples)
        # print(updated_samples)


    def test_star_target(n_components=7, context_dim=2, n_contexts=3, dz=2, n_samples=5):
        from toy_task.GMM.targets.target_factory import get_star_target

        target = get_star_target(n_components, context_dim)
        contexts = target.get_contexts(n_contexts)
        # samples = target.sample(contexts, n_samples)
        # target.visualize(contexts, n_samples=20)
        target.visualize(contexts, filename="general_star_target.pdf")


    def test_funnel_target(context_dim=2, n_contexts=2, n_samples=150, n_components=7):
        from toy_task.GMM.targets.funnel_target import FunnelTarget, get_sig_fn

        target = FunnelTarget(get_sig_fn, context_dim)
        test_ctx = target.get_contexts(n_contexts)
        assert test_ctx.shape == (n_contexts, context_dim)

        # test_samples = target.sample(test_ctx, n_samples)
        # assert test_samples.shape == (n_samples, n_contexts, 10)
        # with ch.enable_grad():
        #     samples = target.update_samples((test_ctx.cuda(), test_samples.cuda()), target.log_prob_tgt, 1e-4, 5)
        # log_probs_component = target.log_prob_tgt(test_ctx, test_samples)
        # assert log_probs_component.shape == (n_samples, n_contexts)
        #
        # test_samples = test_samples.unsqueeze(2).expand(-1, -1, n_components, -1)
        # log_probs = target.log_prob_tgt(test_ctx, test_samples)
        # assert log_probs.shape == (n_samples, n_contexts, n_components)
        #
        # target.visualize(test_ctx, n_samples=20)
        target.visualize(test_ctx, filename="general_funnel_target.pdf")

    test_funnel_target()
