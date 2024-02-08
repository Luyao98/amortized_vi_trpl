#include <projection/SplitDiagMoreProjection.h>

SplitDiagMoreProjection::SplitDiagMoreProjection(uword dim, int max_eval):
    dim(dim),
    max_eval(max_eval)
{}

std::tuple<vec, vec> SplitDiagMoreProjection::forward(double eps_mu, double eps_sig,
                                                      const vec &old_mean, const vec &old_var,
                                                      const vec &target_mean, const vec &target_var) {
    this->eps_mu = eps_mu;
    this->eps_sig = eps_sig;
    this->succ_mu = false;
    this->succ_sig = false;

    this->old_prec = 1.0 / old_var;
    this->old_chol_prec = sqrt(old_prec);
    this->old_mean = old_mean;
    this->old_lin = old_prec % old_mean;

    this->old_dot = dot(old_lin, old_mean);
    this->old_logdet = - 2 * sum(log(old_chol_prec + 1e-25));
    this->kl_const_part = old_logdet - dim;

    this->target_mean = target_mean;
    this->target_prec = 1.0 / target_var;
    this->target_lin = target_prec % target_mean;

    /** mean update **/
    nlopt::opt opt_mean(nlopt::LD_LBFGS, 1);
    opt_mean.set_min_objective([](const std::vector<double> &eta_mu, std::vector<double> &grad, void *instance){
        return ((SplitDiagMoreProjection *) instance)->dual_mean(eta_mu, grad);}, this);
    std::vector<double> opt_eta_mu;
    std::tie(succ_mu, opt_eta_mu) = NlOptUtil::opt_dual_1lp(opt_mean, 0.0, max_eval);
    if (!succ_mu) {
        opt_eta_mu[0] = lp;
        succ_mu = NlOptUtil::valid_despite_failure_1lp(opt_eta_mu, grad);
    }
    if (succ_mu){
        eta_mu = opt_eta_mu[0];

        proj_lin = (target_lin + eta_mu * old_lin) / (eta_mu + 1);
        prec_mu =  (target_prec + eta_mu * old_prec) / (eta_mu + 1);
        cov_mu = 1.0 / prec_mu;
        proj_mean = proj_lin / prec_mu;
    } else {
        throw std::logic_error("NLOPT failure");
    }

    /** covar update **/
    nlopt::opt opt_var(nlopt::LD_LBFGS, 1);
    opt_var.set_min_objective([](const std::vector<double> &eta_sig, std::vector<double> &grad, void *instance){
        return ((SplitDiagMoreProjection *) instance)->dual_cov(eta_sig, grad);}, this);
    std::vector<double> opt_eta_sig;
    std::tie(succ_sig, opt_eta_sig) = NlOptUtil::opt_dual_1lp(opt_var, 0.0, max_eval);
    if (!succ_sig) {
        opt_eta_sig[0] = lp;
        succ_sig = NlOptUtil::valid_despite_failure_1lp(opt_eta_sig, grad);
    }
    if (succ_sig){
       // lp = opt_eta_sig[0];
       // proj_var = (lp + 1.0) / (target_prec + lp * old_prec);
        eta_sig = opt_eta_sig[0];
        proj_var = (eta_sig + 1.0) / (target_prec + eta_sig * old_prec);
        proj_prec = 1 / proj_var;
    } else {
        throw std::logic_error("NLOPT failure");
    }
    return std::make_tuple(proj_mean, proj_var);
}

double SplitDiagMoreProjection::dual_mean(const std::vector<double> &eta_mu, std::vector<double> &grad) {
    this->lp = eta_mu[0] > 0.0 ? eta_mu[0] : 0.0;

    try{
        vec proj_lin = target_lin + lp * old_lin;
        vec proj_mean = proj_lin / (target_prec + lp * old_prec);
        double dual = lp * eps_mu - 0.5 * lp * old_dot ;
        dual += 0.5 * dot(proj_lin, proj_mean);

        grad[0] = eps_mu - 0.5 * accu(square(old_chol_prec % (old_mean - proj_mean)));
        this->grad[0] = grad[0];
        return dual;
    } catch (std::runtime_error &err) {
        grad[0] = -1.0;
        this->grad[0] = grad[0];
        return 1e12;
    }
}

double SplitDiagMoreProjection::dual_cov(const std::vector<double> &eta_sig, std::vector<double> &grad) {
    this->lp = eta_sig[0] > 0.0 ? eta_sig[0] : 0.0;

    try{
        vec proj_cov = (lp + 1) / (target_prec + lp * old_prec);
        vec proj_cov_chol = sqrt(proj_cov);
        double new_logdet = 2 * sum(log(proj_cov_chol) + 1e-25);

        double dual = lp * eps_sig - 0.5 * lp * old_logdet + 0.5 * (lp + 1.0) * new_logdet;

        double trace_term = accu(square(old_chol_prec % proj_cov_chol));
        grad[0] = eps_sig - 0.5 * (kl_const_part - new_logdet + trace_term);
        this->grad[0] = grad[0];
        return dual;
    } catch (std::runtime_error &err) {
        grad[0] = -1.0;
        this->grad[0] = grad[0];
        return 1e12;
    }
}

std::tuple<vec, vec> SplitDiagMoreProjection::backward(const vec &d_means, const vec &d_vars){
    vec deta_mu_dq_target, deta_sig_dQ_target;
    vec deta_mu_dQ_target;

    if (eta_mu > 0) {
        std::tie(deta_mu_dq_target, deta_mu_dQ_target) = last_eta_mu_grad();
    } else {
        deta_mu_dq_target = vec(d_means.size(), fill::zeros);
        deta_mu_dQ_target = vec(d_vars.size(), fill::zeros);
    }
    if (eta_sig > 0) {
        deta_sig_dQ_target = last_eta_sig_grad();
    } else {
        deta_sig_dQ_target = vec(d_vars.size(), fill::zeros);
    }
    vec dq_deta_mu = (old_lin - proj_lin) / (eta_mu + 1);
    vec dQ_mu_deta_mu = (old_prec - prec_mu) / (eta_mu + 1);
    vec dQ_deta_sig = (old_prec - proj_prec) / (eta_sig + 1);

    vec dq = cov_mu %  d_means;
    vec dQ = - proj_var % d_vars % proj_var;
    vec dQ_mu = - cov_mu % d_means % proj_lin % cov_mu;

    double deta_mu = dot(dq, dq_deta_mu) + dot(dQ_mu, dQ_mu_deta_mu);
    double deta_sig = dot(dQ, dQ_deta_sig);

    vec dq_dtarget = deta_mu * deta_mu_dq_target + dq / (eta_mu + 1);
    vec dQ_dtarget = deta_sig * deta_sig_dQ_target + deta_mu * deta_mu_dQ_target
            + dQ / (eta_sig + 1) + dQ_mu / (eta_mu + 1);

    vec d_mu_target = target_prec % dq_dtarget;
    vec d_cov_target = - target_prec % (dq_dtarget % target_mean + dQ_dtarget) % target_prec;

    return std::make_tuple(d_mu_target, d_cov_target);
}



std::tuple<vec, vec> SplitDiagMoreProjection::last_eta_mu_grad() const {
    vec dQ_mu_deta_mu = (old_prec - prec_mu) / (eta_mu + 1);
    vec dq_deta_mu = (old_lin - proj_lin) / (eta_mu + 1);

    vec dmean_dq = 2 * cov_mu % (old_prec % proj_mean - old_lin);
    vec dmean_dQ_mu = cov_mu % (2 * proj_lin % (old_lin  - old_prec % cov_mu % proj_lin)) % cov_mu;

    double dmean_deta_mu = dot(dmean_dq , dq_deta_mu) + dot(dmean_dQ_mu, dQ_mu_deta_mu);
    vec lhs_mu = dmean_dq / (eta_mu + 1);
    vec lhs_sig = dmean_dQ_mu / (eta_mu + 1);

    return std::make_tuple(lhs_mu / - dmean_deta_mu, lhs_sig / - dmean_deta_mu);
}

vec SplitDiagMoreProjection::last_eta_sig_grad() const {
    vec dQ_deta_sig = (old_prec - proj_prec) / (eta_sig + 1);
    vec dvar_dQ = - proj_var % old_prec % proj_var + proj_var;
    double dvar_deta_sig = dot(dvar_dQ, dQ_deta_sig);
    vec lhs = dvar_dQ / (eta_sig + 1);
    return lhs / - dvar_deta_sig;
}