#ifndef IPP_BATCHEDPROJECTION_H
#define IPP_BATCHEDPROJECTION_H

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>
#include <projection/MoreProjection.h>
#include <cblas.h>

using namespace arma;

class BatchedProjection{

public:

    BatchedProjection(uword batch_size, uword dim, bool eec, bool constrain_entropy, int max_eval);

    std::tuple<mat, cube> forward(const vec &epss, const vec &betas,
                                  const mat &old_means, const cube &old_covars,
                                  const mat &target_means, const cube &target_covars);

    std::tuple<mat, cube> backward(const mat &d_means, const cube &d_covs);

private:


    double kl(const vec& m1, const mat& cc1, const vec& m2, const mat& cc2) const;
    double entropy(const mat& cc) const;
    double entropy_const_part;
    std::vector<MoreProjection> projectors;
    uword batch_size, dim;
    std::vector<bool> projection_applied;
    bool eec, constrain_entropy;


};

#endif //IPP_BATCHEDPROJECTION_H
