#include <projection/MoreProjection.h>

MoreProjection::MoreProjection(uword dim, bool eec, bool constrain_entropy, int max_eval) :
    dim(dim),
    eec(eec),
    constrain_entropy(constrain_entropy),
    max_eval(max_eval)
{
    dual_const_part = dim * log(2 * M_PI);
    entropy_const_part = 0.5 * (dual_const_part + dim);
    omega_offset = 1.0;  // Default value, might change due to rescaling!
}

std::tuple<vec, mat> MoreProjection::forward(double eps, double beta,
                                             const vec  &old_mean, const mat &old_covar,
                                             const vec &target_mean, const mat &target_covar){
    this->eps = eps;
    this->beta = constrain_entropy ? beta : 0.0;
    succ = false;

    /** Prepare **/
    this->old_mean = old_mean;
    old_precision = inv_sympd(old_covar);
    old_lin = old_precision * old_mean;
    old_chol_precision_t = chol(old_precision, "lower").t();

    this->target_mean = target_mean;
    target_precision = inv_sympd(target_covar);
    target_lin = target_precision * target_mean;

    double old_logdet = - 2 * sum(log(diagvec(old_chol_precision_t) + 1e-25));
    old_term = -0.5 * (dot(old_lin, old_mean) + old_logdet + dual_const_part);
    kl_const_part = old_logdet - dim;

    /** Otpimize **/
    nlopt::opt opt(nlopt::LD_LBFGS, 2);

    opt.set_min_objective([](const std::vector<double> &eta_omega, std::vector<double> &grad, void *instance){
        return ((MoreProjection *) instance)->dual(eta_omega, grad);}, this);

    std::vector<double> opt_eta_omega;

    std::tie(succ, opt_eta_omega) =
            NlOptUtil::opt_dual_2lp(opt, 0.0, eec ? -1e12 : 0.0, max_eval);
    opt_eta_omega[1] = constrain_entropy ? opt_eta_omega[1] : 0.0;

    if (!succ) {
        opt_eta_omega[0] = eta;
        opt_eta_omega[1] = constrain_entropy ? omega : 0.0;
        succ = NlOptUtil::valid_despite_failure_2lp(opt_eta_omega, grad);
    }

    /** Post process**/
    std::tuple<vec, mat> res;
    if (succ) {
        eta = opt_eta_omega[0];
        omega = constrain_entropy ? opt_eta_omega[1] : 0.0;

        projected_lin = (eta * old_lin + target_lin) / (eta + omega + omega_offset);
        projected_precision = (eta * old_precision + target_precision) / (eta + omega + omega_offset);
        projected_covar = inv_sympd((projected_precision));
        projected_mean = projected_covar * projected_lin;
        res = std::make_tuple(projected_mean, projected_covar);
    } else{
        throw std::logic_error("NLOPT failure");
        //res = std::make_tuple(vec(dim, fill::zeros), mat(dim, dim, fill::zeros));
    }
    return res;
}

std::tuple<vec, mat> MoreProjection::backward(const vec &d_mean, const mat &d_cov) {
    /** takes derivatives of loss w.r.t to projected mean and covariance and propagates back through optimization
      yielding derivatives of loss w.r.t to target mean and covariance **/
    if (!succ){
        throw std::exception();
    }
    /** Prepare **/

    vec deta_dq_target, domega_dq_target;
    mat deta_dQ_target, domega_dQ_target;
    std::tie(deta_dq_target, deta_dQ_target, domega_dq_target, domega_dQ_target) = last_eo_grad();

    double eo = omega + omega_offset + eta;
    double eo_squared = eo * eo;
    vec dq_deta = ((omega + omega_offset) * old_lin - target_lin) / eo_squared;
    mat dQ_deta = ((omega + omega_offset) * old_precision - target_precision) / eo_squared;
    vec dq_domega = - (eta * old_lin + target_lin) / eo_squared;
    mat dQ_domega = - (eta * old_precision + target_precision) / eo_squared;

    vec d_q = projected_covar * d_mean;
    mat tmp = d_mean * projected_lin.t();
    mat d_Q = - projected_covar * (0.5 * tmp + 0.5 * tmp.t() + d_cov) * projected_covar;

    double d_eta = dot(d_q, dq_deta) + trace(d_Q * dQ_deta);
    double d_omega = dot(d_q, dq_domega) + trace(d_Q * dQ_domega);

    vec d_q_target = d_eta * deta_dq_target + d_omega * domega_dq_target + d_q / eo;
    mat d_Q_target = d_eta * deta_dQ_target + d_omega * domega_dQ_target + d_Q / eo;

    vec d_mu_target = target_precision * d_q_target;
    tmp = d_q_target * target_mean.t();
    mat d_cov_target = - target_precision * (0.5 * tmp + 0.5 * tmp.t() + d_Q_target) * target_precision;

    return std::make_tuple(d_mu_target, d_cov_target);
}

double MoreProjection::dual(std::vector<double> const &eta_omega, std::vector<double> &grad){
    eta = eta_omega[0] > 0.0 ? eta_omega[0] : 0.0;
    omega = eta_omega[1] > 0.0 && constrain_entropy ? eta_omega[1] : 0.0;
    double omega_off = omega + omega_offset;

    vec new_lin = (eta * old_lin + target_lin) / (eta + omega_off);
    mat new_quad = (eta * old_precision + target_precision) / (eta + omega_off);
    try {
        /** dual **/
        mat new_covar = inv_sympd(new_quad);
        mat new_chol_covar = chol(new_covar, "lower");

        vec new_mean = new_covar * new_lin;
        double new_logdet = 2 * sum(log(diagvec(new_chol_covar) + 1e-25));

        double dual = eta * eps - omega * beta + eta * old_term;
        dual += 0.5 * (eta + omega_off) * (dual_const_part + new_logdet + dot(new_lin, new_mean));

        /** gradient **/
        vec diff = old_mean - new_mean;
        double trace_term = accu(square(old_chol_precision_t * new_chol_covar));
        double kl = 0.5 * (sum(square(old_chol_precision_t * diff)) + kl_const_part - new_logdet + trace_term);

        grad[0] = eps - kl;
        grad[1] = constrain_entropy ? entropy_const_part + 0.5 * new_logdet - beta: 0.0;
        this->grad[0] = grad[0];
        this->grad[1] = grad[1];

        return dual;
    } catch (std::runtime_error &err) {
        grad[0] = -1.0;
        grad[1] = 0.0;
        this->grad[0] = grad[0];
        this->grad[1] = grad[1];
        return 1e12;
    }
}

std::tuple<vec, mat, vec, mat> MoreProjection::last_eo_grad() const {
    /** case 1, neither active **/
    if(eta == 0.0 && omega == 0.0){
        return std::make_tuple(vec(dim, fill::zeros), mat(dim, dim, fill::zeros),
                               vec(dim, fill::zeros), mat(dim, dim, fill::zeros));

    /** case 2, entropy constraint active **/
    } else if(eta == 0.0 && omega != 0.0) {
        mat domega_dQ = projected_covar / dim;
        return std::make_tuple(vec(dim, fill::zeros), mat(dim, dim, fill::zeros),
                               vec(dim, fill::zeros), domega_dQ);

    /** case 3, kl constraint active **/
    }  else if(eta > 0.0 && omega == 0.0){
        vec dq_deta = ((omega + omega_offset) * old_lin - target_lin) / (eta + omega + omega_offset);
        mat dQ_deta = ((omega + omega_offset) * old_precision - target_precision) / (eta + omega + omega_offset);

        vec f2_dq = 2 * projected_covar * (old_precision * projected_mean - old_lin);
        mat tmp1 = projected_lin * old_lin.t();
        mat tmp2 = old_precision * projected_covar * (projected_lin * projected_lin.t());
        mat tmp = mat(dim, dim, fill::eye) + (-old_precision + tmp1 + tmp1.t() - tmp2 - tmp2.t()) * projected_covar;
        mat f2_dQ = projected_covar * tmp;

        double c = - 1  / (trace(f2_dQ * dQ_deta) + dot(f2_dq, dq_deta));
        return std::make_tuple(c * f2_dq, c * f2_dQ, vec(dim, fill::zeros), mat(dim, dim, fill::zeros));

    /** case 4, both active **/
    } else if(eta > 0.0 && omega != 0.0) {
        vec dq_deta = ((omega + omega_offset) * old_lin - target_lin) / (eta + omega + omega_offset);
        mat dQ_deta = ((omega + omega_offset) * old_precision - target_precision) / (eta + omega + omega_offset);
        vec dq_domega = - projected_lin;
        mat dQ_domega = - projected_precision;

        vec f2_dq = 2 * projected_covar * (old_precision * projected_mean - old_lin);

        mat tmp1 = projected_lin * old_lin.t();
        mat tmp2 = old_precision * projected_covar * (projected_lin * projected_lin.t());
        mat tmp = mat(dim, dim, fill::eye) + (-old_precision + tmp1 + tmp1.t() - tmp2 - tmp2.t()) * projected_covar;
        mat f2_dQ = projected_covar * tmp;

        double f2_deta = dot(f2_dq, dq_deta) + trace(f2_dQ * dQ_deta);
        double f2_domega = dot(f2_dq, dq_domega) + trace(f2_dQ * dQ_domega);

        double tr_logdet_deta = trace(projected_covar * dQ_deta);
        double tr_logdet_domega = trace(projected_covar * dQ_domega);

        double lhs_eta = f2_deta - (tr_logdet_deta / tr_logdet_domega) * f2_domega;
        double lhs_omega = f2_domega - (tr_logdet_domega / tr_logdet_deta) * f2_deta;

        vec rhs_q = - f2_dq;
        vec deta_dq = rhs_q / lhs_eta;
        vec domega_dq = rhs_q / lhs_omega;
        mat deta_dQ = ((f2_domega / tr_logdet_domega) * projected_covar - f2_dQ) / lhs_eta;
        mat domega_dQ = ((f2_deta / tr_logdet_deta) * projected_covar - f2_dQ) / lhs_omega;

        return std::make_tuple(deta_dq, deta_dQ, domega_dq, domega_dQ);
    } else {
        throw std::exception();
    }
}
