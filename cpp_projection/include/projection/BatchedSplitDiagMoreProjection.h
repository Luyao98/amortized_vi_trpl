#ifndef IPP_BATCHEDSPLITDIAGMOREPROJECTION_H
#define IPP_BATCHEDSPLITDIAGMOREPROJECTION_H

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>
#include <projection/SplitDiagMoreProjection.h>
#include <cblas.h>

using namespace arma;

class BatchedSplitDiagMoreProjection{

public:

    BatchedSplitDiagMoreProjection(uword batch_size, uword dim, int max_eval);

    std::tuple<mat, mat> forward(const vec &epss_mu, const vec &epss_sig,
                                  const mat &old_means, const mat &old_vars,
                                  const mat &target_means, const mat &target_vars);

    std::tuple<mat, mat> backward(const mat &d_means, const mat &d_vars);

private:

    double kl_diag_mean(const vec& m1, const vec& m2, const vec& cc2) const;
    double kl_diag_var(const vec& cc1, const vec& cc2) const;
    std::vector<SplitDiagMoreProjection> projectors;
    uword batch_size, dim;
    std::vector<bool> projection_applied;

};

#endif //IPP_BATCHEDSPLITDIAGMOREPROJECTION_H
