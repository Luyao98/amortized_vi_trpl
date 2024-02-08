#include <projection/BatchedProjection.h>
#include <chrono>
using namespace std::chrono;


BatchedProjection::BatchedProjection(uword batch_size, uword dim, bool eec, bool constrain_entropy, int max_eval) :
    batch_size(batch_size),
    dim(dim),
    eec(eec),
    constrain_entropy(constrain_entropy){
    for (int i = 0; i < batch_size; ++i) {
        projectors.emplace_back(MoreProjection(dim, eec, constrain_entropy, max_eval));
        projection_applied.emplace_back(false);
    }
    entropy_const_part = 0.5 * (dim * log(2 * M_PI * M_E)); 

    openblas_set_num_threads(1);
}

std::tuple<mat, cube> BatchedProjection::forward(const vec &epss, const vec &betas,
                                                 const mat &old_means, const cube &old_covars,
                                                 const mat &target_means, const cube &target_covars) {

    mat means(size(old_means));
    cube covs(size(old_covars));
    auto start = high_resolution_clock::now();
    bool failed = false;
    std::stringstream stst;

#pragma omp parallel for default(none) schedule(static) shared(epss, betas, old_means, old_covars, target_means, target_covars, means, covs, failed, stst)
    for (int i = 0; i < batch_size; ++i) {
        double eps = epss.at(i);
        double beta = betas.at(i);
        const vec &old_mean = old_means.col(i);
        const mat &old_cov = old_covars.slice(i);
        const vec &target_mean = target_means.col(i);
        const mat &target_cov = target_covars.slice(i);


        if (eec) {
            vec mean;
            mat cov;
            try {
                std::tie(mean, cov) = projectors[i].forward(eps, beta, old_mean, old_cov, target_mean, target_cov);
                means.col(i) = mean;
                covs.slice(i) = cov;
                projection_applied.at(i) = true;
            } catch (std::logic_error &e) {
                stst << "Failure during projection " << i << ": " << e.what() << " ";
                failed = true;
            }
        } else {
            mat occ = chol(old_cov, "lower");
            mat tcc = chol(target_cov, "lower");
            double kl_ = kl(target_mean, tcc, old_mean, occ);
            double entropy_ = entropy(tcc);
            if (kl_ <= eps && (entropy_ >= beta || !constrain_entropy)) {
                means.col(i) = target_mean;
                covs.slice(i) = target_cov;
                projection_applied.at(i) = false;
            } else {
                vec mean;
                mat cov;
                try {
                    std::tie(mean, cov) = projectors[i].forward(eps, beta, old_mean, old_cov, target_mean, target_cov);
                    means.col(i) = mean;
                    covs.slice(i) = cov;
                    projection_applied.at(i) = true;
                } catch (std::logic_error &e) {
                    stst << "Failure during projection " << i << ": " << e.what() << " ";
                    failed = true;
                }

            }
        }
    }
    if (failed) {
        throw std::invalid_argument(stst.str());
    }
    return std::make_tuple(means, covs);
}

std::tuple<mat, cube> BatchedProjection::backward(const mat &d_means, const cube &d_covs) {
    mat d_means_target(size(d_means));
    cube d_covs_target(size(d_covs));
#pragma omp parallel for default(none) schedule(static) shared(d_means, d_covs, d_means_target, d_covs_target)
    for (int i = 0; i < batch_size; ++i) {
        vec d_mean = d_means.col(i);
        mat d_cov = d_covs.slice(i);

        if (!projection_applied.at(i)) {
            d_means_target.col(i) = d_mean;
            d_covs_target.slice(i) = d_cov;
        } else {
            vec d_mean_target;
            mat d_cov_target;
            std::tie(d_mean_target, d_cov_target) = projectors[i].backward(d_mean, d_cov);
            d_means_target.col(i) = d_mean_target;
            d_covs_target.slice(i) = d_cov_target;
        }
    }
    return std::make_tuple(d_means_target, d_covs_target);
}


double BatchedProjection::kl(const vec& m1, const mat& cc1, const vec& m2, const mat& cc2) const {
    mat cc2_inv_t = inv(cc2);
    double logdet_term = 2 * (sum(log(diagvec(cc2) + 1e-25)) - sum(log(diagvec(cc1) + 1e-25)));
    double trace_term = accu(square(cc2_inv_t * cc1));
    double mahal_term = sum(square(cc2_inv_t * (m2 - m1)));
    double kl = 0.5 * (logdet_term + trace_term + mahal_term - dim);
    return kl;
}

double BatchedProjection::entropy(const mat& cc) const {
    return entropy_const_part + sum(log(diagvec(cc) + 1e-25));
}
