#include <projection/CovOnlyMoreProjection.h>

CovOnlyMoreProjection::CovOnlyMoreProjection(uword dim, int max_eval) :
    dim(dim),
    max_eval(max_eval)
{
    omega_offset = 1.0;  // Default value, might change due to rescaling!
}
//mat CovOnlyMoreProjection::forward(double eps, const mat &old_covar, const mat &target_covar){
mat CovOnlyMoreProjection::forward(double eps, const mat &old_chol, const mat &target_covar){
    this->eps = eps;
    succ = false;

//    std::cout << "Input" << std::endl;
//    std::cout << target_covar << std::endl;
//    std::cout << old_chol << std::endl;

    /** Prepare **/
    mat I = mat(dim, dim, fill::eye );

    old_chol_L = trimatl(old_chol);
    mat old_covar = old_chol_L * old_chol_L.t();
    old_precision = solve(old_covar, I);

    target_precision = solve(target_covar, I);

    old_logdet = 2 * sum(log(diagvec(old_chol_L) + 1e-25));
    kl_const_part = old_logdet - dim;

    /** Optimize **/
    nlopt::opt opt(nlopt::LD_LBFGS, 1);

    opt.set_min_objective([](const std::vector<double> &eta, std::vector<double> &grad, void *instance){
        return ((CovOnlyMoreProjection *) instance)->dual(eta, grad);}, this);

    std::vector<double> opt_eta;
    std::tie(succ, opt_eta) = NlOptUtil::opt_dual_1lp(opt, 0.0, max_eval);

    if (!succ) {
        opt_eta[0] = eta;
        succ = NlOptUtil::valid_despite_failure_1lp(opt_eta, grad);
    }

    /** Post process**/
    if (succ) {
        eta = opt_eta[0];
        projected_precision = (eta * old_precision + target_precision) / (eta + omega_offset);
        projected_covar = solve(projected_precision, mat(dim, dim, fill::eye ));
    } else{
//        std::cout << "Failed" << std::endl;
//        throw std::logic_error("NLOPT failure");
        mat res = mat(dim, dim);
        res.fill(datum::nan);
        return res;
        //res = std::make_tuple(vec(dim, fill::zeros), mat(dim, dim, fill::zeros));
    }
//    return projected_precision;
    return projected_covar;
}

mat CovOnlyMoreProjection::backward(const mat &d_cov) {
   # /** takes derivatives of loss w.r.t to projected mean and covariance and propagates back through optimization
    #  yielding derivatives of loss w.r.t to target mean and covariance **/
    if (!succ){
//        throw std::exception();
        return mat(dim, dim, fill::zeros);
    }
    /** Prepare **/

    mat deta_dQ_target = last_eo_grad();

    double eo = omega_offset + eta;
    double eo_squared = eo * eo;

    mat dQ_deta = (omega_offset * old_precision - target_precision) / eo_squared;

    mat d_Q = - projected_covar * d_cov * projected_covar;

    double d_eta = trace(d_Q * dQ_deta);

    mat d_Q_target = d_eta * deta_dQ_target + d_Q / eo;

    mat d_cov_target = - target_precision * d_Q_target * target_precision;

//    grad l * l.t()
//TODO Grad for cholesky input
    return d_cov_target;
}

double CovOnlyMoreProjection::dual(std::vector<double> const &eta_omega, std::vector<double> &grad){
    eta = eta_omega[0] > 0.0 ? eta_omega[0] : 0.0;
    mat new_precision = (eta * old_precision + target_precision) / (eta + omega_offset);
    try {
        /** dual **/
        mat I = mat( size(new_precision), fill::eye );
        mat new_covar = solve(new_precision, I);
//        mat new_covar = inv_sympd(new_precision);
        mat new_chol_covar = trimatl(chol(new_covar, "lower"));
        double new_logdet = 2 * sum(log(diagvec(new_chol_covar) + 1e-25));

        double dual = eta * eps - 0.5 * eta * old_logdet;
        dual += 0.5 * (eta + omega_offset) * new_logdet;
        /** gradient **/
        double trace_term = accu(square(solve(old_chol_L, new_chol_covar)));
        double kl = 0.5 * (kl_const_part - new_logdet + trace_term);
        grad[0] = eps - kl;
        this->grad[0] = grad[0];

        return dual;
    } catch (std::runtime_error &err) {
//        std::cout << "Catch" << std::endl;
        grad[0] = -1.0;
        this->grad[0] = grad[0];
        return 1e12;
    }
}

mat CovOnlyMoreProjection::last_eo_grad() const {
    /** case 1, neither active **/
    if(eta == 0.0){
        return mat(dim, dim, fill::zeros);

    /** case 3, kl constraint active **/
    }  else if(eta > 0.0){
        mat dQ_deta = (omega_offset * old_precision - target_precision) / (eta + omega_offset);

        mat tmp = mat(dim, dim, fill::eye) - old_precision * projected_covar;
        mat f2_dQ = projected_covar * tmp;

        double c = - 1  / trace(f2_dQ * dQ_deta);
        return c * f2_dQ;

    } else {
        throw std::exception();
    }
}
