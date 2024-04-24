#ifndef IPP_BATCHEDCOVONLYPROJECTION_H
#define IPP_BATCHEDCOVONLYPROJECTION_H

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>
#include <projection/CovOnlyMoreProjection.h>
#include <cblas.h>

using namespace arma;

class BatchedCovOnlyProjection{

public:

    BatchedCovOnlyProjection(uword batch_size, uword dim, int max_eval);

    cube forward(const vec &epss, const cube &old_chols, const cube &target_chols, const cube &target_covars);
    cube backward(const cube &d_covs);

private:

    double kl_cov_only(const mat& cc1, const mat& cc2) const;
    std::vector<CovOnlyMoreProjection> projectors;
    uword batch_size, dim;
    std::vector<bool> projection_applied;

};

#endif //IPP_BATCHEDCOVONLYPROJECTION_H
