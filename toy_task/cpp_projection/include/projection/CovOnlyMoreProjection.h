#ifndef CPP_COVONLYMOREPROJECTION_H
#define CPP_COVONLYMOREPROJECTION_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>


using namespace arma;

class CovOnlyMoreProjection{

public:
    CovOnlyMoreProjection(uword dim, int max_eval);

    mat forward(double eps, const mat &old_chol, const mat &target_covar);

    mat backward(const mat &d_cov);

    double get_last_eta() const { return eta;};
    bool was_succ() const {return succ;}

private:

    mat last_eo_grad() const;

    double dual(std::vector<double> const &eta_omega, std::vector<double> &grad);

    double eps, omega_offset;
    bool succ;
    uword dim;
    double eta=1;
    std::vector<double> grad = std::vector<double>(1, 10);

    int max_eval;
    double old_logdet, old_term, kl_const_part;

    mat old_precision, old_chol_precision_t, target_precision, projected_covar, projected_precision, old_chol_L, old_covar;

};
#endif //CPP_COVONLYMOREPROJECTION_H