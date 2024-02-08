#include <projection/BatchedSplitDiagMoreProjection.h>
#include <chrono>
using namespace std::chrono;

BatchedSplitDiagMoreProjection::BatchedSplitDiagMoreProjection(uword batch_size, uword dim, int max_eval) :
    batch_size(batch_size),
    dim(dim){
    for (int i = 0; i < batch_size; ++i) {
        projectors.emplace_back(SplitDiagMoreProjection(dim, max_eval));
        projection_applied.emplace_back(false);
    }
    openblas_set_num_threads(1);
}

std::tuple<mat, mat> BatchedSplitDiagMoreProjection::forward(const vec &epss_mu, const vec &epss_sig,
                                                             const mat &old_means, const mat &old_vars,
                                                             const mat &target_means, const mat &target_vars) {
    auto start = high_resolution_clock::now();
    mat means(size(old_means));
    mat vars(size(old_vars));
    bool failed = false;
    std::stringstream stst;

    #pragma omp parallel for default(none) schedule(static) shared(epss_mu, epss_sig, old_means, old_vars, target_means, target_vars, means, vars, failed, stst)
    for (int i = 0; i < batch_size; ++i) {
        double eps_mu = epss_mu.at(i);
        double eps_sig = epss_sig.at(i);
        const vec &old_mean = old_means.col(i);
        const mat &old_var = old_vars.col(i);
        const vec &target_mean = target_means.col(i);
        const mat &target_var = target_vars.col(i);

        vec occ = sqrt(old_var);
        vec tcc = sqrt(target_var);
        double kl_mean = kl_diag_mean(target_mean, old_mean, occ);
        double kl_cov = kl_diag_var(tcc, occ);
        if (kl_mean <= eps_mu && kl_cov <= kl_cov) {
            means.col(i) = target_mean;
            vars.col(i) = target_var;
            projection_applied.at(i) = false;
        } else {
           // std::cout << "else" << std::endl;
            vec mean;
            vec var;
            try {
                std::tie(mean, var) = projectors[i].forward(eps_mu, eps_sig, old_mean, old_var, target_mean, target_var);
                means.col(i) = mean;
                vars.col(i) = var;
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
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    //cout << "time " << duration.count() / 1000000.0 << endl;
    return std::make_tuple(means, vars);
}


std::tuple<mat, mat> BatchedSplitDiagMoreProjection::backward(const mat &d_means, const mat &d_vars) {
    mat d_mus_target(size(d_means));
    mat d_vars_target(size(d_vars));

    #pragma omp parallel for default(none) schedule(static) shared(d_means, d_vars, d_mus_target, d_vars_target)
    for (int i = 0; i < batch_size; ++i) {
        vec d_mean = d_means.col(i);
        vec d_cov = d_vars.col(i);

        if (!projection_applied.at(i)) {
            d_mus_target.col(i) = d_mean;
            d_vars_target.col(i) = d_cov;
        } else {
            vec d_mean_target;
            vec d_cov_target;
            std::tie(d_mean_target, d_cov_target) = projectors[i].backward(d_mean, d_cov);
            d_mus_target.col(i) = d_mean_target;
            d_vars_target.col(i) = d_cov_target;
        }
    }
    return std::make_tuple(d_mus_target, d_vars_target);
}


double BatchedSplitDiagMoreProjection::kl_diag_mean(const vec &m1, const vec &m2, const vec &cc2) const {
    vec cc_inv = 1 / cc2;
    return 0.5 * sum(square(cc_inv % (m2 - m1)));

}

double BatchedSplitDiagMoreProjection::kl_diag_var(const vec &cc1, const vec &cc2) const {
    vec cc2_inv_t = 1.0 / cc2;
    double logdet_term = 2 * (sum(log(cc2 + 1e-25)) - sum(log(cc1 + 1e-25)));
    double trace_term = sum(square(cc2_inv_t % cc1));
    double kl = 0.5 * (logdet_term + trace_term - dim);
    return kl;
}


