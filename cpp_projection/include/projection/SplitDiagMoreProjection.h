#ifndef CPP_SPLITDIAGMOREPROJECTION_H
#define CPP_SPLITDIAGMOREPROJECTION_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>


using namespace arma;

class SplitDiagMoreProjection{

public:
    SplitDiagMoreProjection(uword dim, int max_eval);

    std::tuple<vec, vec> forward(double eps_mu, double eps_sigma,
                                 const vec  &old_mean, const vec &old_var,
                                 const vec &target_mean, const vec &target_var);

    std::tuple<vec, vec> backward(const vec &d_means, const vec &d_vars);

    bool was_succ() const {return succ_mu && succ_sig;};

private:

    std::tuple<vec, vec> last_eta_mu_grad() const;
    vec last_eta_sig_grad() const;


    double dual_mean(std::vector<double> const &eta_mu, std::vector<double> &grad);
    double dual_cov(std::vector<double> const &eta_sig, std::vector<double> &grad);

    double eps_mu, eps_sig;
    double eta_mu, eta_sig;

    bool succ_mu, succ_sig;
    uword dim, eta_inc_ct;
    double lp=1.0;
    std::vector<double> grad = std::vector<double>(1.0, 10);

    int max_eval;
    double old_dot, old_logdet, kl_const_part;

    vec old_mean, old_lin, old_var, old_prec, old_chol_prec;
    vec target_mean, target_lin, target_prec;
    vec proj_mean, proj_lin, prec_mu, cov_mu, proj_var, proj_prec;
};
#endif //CPP_SPLITDIAGMOREPROJECTION_H
