#include <projection/BatchedCovOnlyProjection.h>

BatchedCovOnlyProjection::BatchedCovOnlyProjection(uword batch_size, uword dim, int max_eval) :
    batch_size(batch_size),
    dim(dim){
    for (int i = 0; i < batch_size; ++i) {
        projectors.emplace_back(CovOnlyMoreProjection(dim, max_eval));
        projection_applied.emplace_back(false);
    }
    openblas_set_num_threads(1);
}

cube BatchedCovOnlyProjection::forward(const vec &epss, const cube &old_chols, const cube &target_chols, const cube &target_covars) {

    cube covs(size(old_chols));
    bool failed = false;
    std::stringstream stst;

// TODO
//    std::cout << size(old_chols) << std::endl;
//    std::cout << size(old_chols)[0] << std::endl;

    #pragma omp parallel for default(none) schedule(static) shared(epss, old_chols, target_chols, target_covars, covs, failed, stst)
    for (int i = 0; i < batch_size; ++i) {
        double eps = epss.at(i);
        const mat &old_chol = old_chols.slice(i);
        const mat &target_chol = target_chols.slice(i);
        const mat &target_cov = target_covars.slice(i);

//  TODO: transpose as python is row major and armadillo column major, hence cholesky would be triu not tril
//        mat occ = chol(old_cov, "lower");
//        mat tcc = chol(target_cov, "lower");
        mat occ = trimatl(old_chol.t());
        mat tcc = trimatl(target_chol.t());
        double kl_ = kl_cov_only(tcc, occ);

        if (kl_ <= eps) {
            covs.slice(i) = target_cov;
            projection_applied.at(i) = false;
        } else {
            try {
//                mat cov = projectors[i].forward(eps, old_chol, target_cov);
                mat cov = projectors[i].forward(eps, occ, target_cov);
                covs.slice(i) = cov;
                projection_applied.at(i) = true;
            } catch (std::logic_error &e) {
                stst << "Failure during projection " << i << ": " << e.what() << " ";
                failed = true;
            }
        }
    }
    if (failed) {
        throw std::invalid_argument(stst.str());
    }
    return covs;
}

cube BatchedCovOnlyProjection::backward(const cube &d_covs) {
    cube d_covs_target(size(d_covs));

    #pragma omp parallel for default(none) schedule(static) shared(d_covs, d_covs_target)
    for (int i = 0; i < batch_size; ++i) {
        mat d_cov = d_covs.slice(i);

        d_covs_target.slice(i) = projection_applied.at(i) ? projectors[i].backward(d_cov) : d_cov;
    }
    return d_covs_target;
}


double BatchedCovOnlyProjection::kl_cov_only(const mat &cc1, const mat &cc2) const {
    mat cc2_inv_t = inv(cc2);
    double logdet_term = 2 * (sum(log(diagvec(cc2) + 1e-25)) - sum(log(diagvec(cc1) + 1e-25)));
    double trace_term = accu(square(cc2_inv_t * cc1));
    double kl = 0.5 * (logdet_term + trace_term - dim);
    return kl;
}

