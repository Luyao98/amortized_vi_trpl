#include <pybind11/pybind11.h>
#include <PyArmaConverter.h>

#include <projection/MoreProjection.h>
#include <projection/BatchedProjection.h>

#include <projection/DiagCovOnlyMoreProjection.h>
#include <projection/BatchedDiagCovOnlyProjection.h>

#include <projection/SplitDiagMoreProjection.h>
#include <projection/BatchedSplitDiagMoreProjection.h>

#include <projection/CovOnlyMoreProjection.h>
#include <projection/BatchedCovOnlyProjection.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_projection, p){
    /* ------------------------------------------------------------------------------
     MORE PROJECTION
     --------------------------------------------------------------------------------*/
    py::class_<MoreProjection> mp(p, "MoreProjection");

    mp.def(py::init([](uword dim, bool eec, bool constrain_entropy, int max_eval){
        return new MoreProjection(dim, eec, constrain_entropy, max_eval);}),
            py::arg("dim"), py::arg("eec"), py::arg("constrain_entropy"), py::arg("max_eval") = 100);

    mp.def("forward", [](MoreProjection* obj, double eps, double beta,
                                  dpy_arr old_mean, dpy_arr old_covar,
                                  dpy_arr target_mean, dpy_arr target_covar){
               return from_mat<double>(obj->forward(eps, beta, to_vec<double>(old_mean), to_mat<double>(old_covar),
                                                    to_vec<double>(target_mean), to_mat<double>(target_covar)));},
    py::arg("eps"), py::arg("beta"), py::arg("old_mean"),
    py::arg("old_covar"), py::arg("target_mean"), py::arg("target_covar"));

    mp.def("backward", [](MoreProjection* obj, dpy_arr dl_dmu_projected, dpy_arr dl_dcovar_projected){
        vec dl_dmu_target;
        mat dl_dcovar_target;
        std::tie(dl_dmu_target, dl_dcovar_target) = obj->backward(to_vec<double>(dl_dmu_projected),
                                                                        to_mat<double>(dl_dcovar_projected));
        return std::make_tuple(from_mat<double>(dl_dmu_target), from_mat<double>(dl_dcovar_target));},
    py::arg("dl_dmu_projected"), py::arg("dl_dcovar_projected"));

    mp.def_property_readonly("last_eta", &MoreProjection::get_last_eta);
    mp.def_property_readonly("last_omega", &MoreProjection::get_last_omega);
    mp.def_property_readonly("was_succ", &MoreProjection::was_succ);

    /* ------------------------------------------------------------------------------
    DIAG COVAR ONLY PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<DiagCovOnlyMoreProjection> dcop(p, "DiagCovOnlyMoreProjection");

    dcop.def(py::init([](uword dim, int max_eval){return new DiagCovOnlyMoreProjection(dim, max_eval);}),
           py::arg("dim"), py::arg("max_eval") = 100);

    dcop.def("forward", [](DiagCovOnlyMoreProjection* obj, double eps, dpy_arr old_covar, dpy_arr target_covar){
               return from_mat<double>(obj->forward(eps, to_vec<double>(old_covar), to_vec<double>(target_covar)));},
           py::arg("eps"),py::arg("old_covar"), py::arg("target_covar"));

    dcop.def("backward", [](DiagCovOnlyMoreProjection* obj, dpy_arr dl_dcovar_projected){
               return from_mat<double>(obj->backward(to_vec<double>(dl_dcovar_projected)));},
      py::arg("dl_dcovar_projected"));

    dcop.def_property_readonly("last_eta", &DiagCovOnlyMoreProjection::get_last_eta);
    dcop.def_property_readonly("was_succ", &DiagCovOnlyMoreProjection::was_succ);

    /* ------------------------------------------------------------------------------
    COVAR ONLY PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<CovOnlyMoreProjection> cop(p, "CovOnlyMoreProjection");

    cop.def(py::init([](uword dim, int max_eval){return new CovOnlyMoreProjection(dim, max_eval);}),
             py::arg("dim"), py::arg("max_eval") = 100);

    cop.def("forward", [](CovOnlyMoreProjection* obj, double eps, dpy_arr old_chol, dpy_arr target_covar){
                 return from_mat<double>(obj->forward(eps, to_mat<double>(old_chol), to_mat<double>(target_covar)));},
             py::arg("eps"),py::arg("old_chol"),py::arg("target_covar"));

    cop.def("backward", [](CovOnlyMoreProjection* obj, dpy_arr dl_dcovar_projected){
                 return from_mat<double>(obj->backward(to_mat<double>(dl_dcovar_projected)));},
             py::arg("dl_dcovar_projected"));

    cop.def_property_readonly("last_eta", &CovOnlyMoreProjection::get_last_eta);
    cop.def_property_readonly("was_succ", &CovOnlyMoreProjection::was_succ);

    /* ------------------------------------------------------------------------------
    SPLIT DIAG COVAR PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<SplitDiagMoreProjection> sdcmp(p, "SplitDiagMoreProjection");

    sdcmp.def(py::init([](uword dim, int max_eval){return new SplitDiagMoreProjection(dim, max_eval);}),
           py::arg("dim"), py::arg("max_eval") = 100);

    sdcmp.def("forward", [](SplitDiagMoreProjection* obj, double eps_mu, double eps_sig,
                                   dpy_arr old_mean, dpy_arr old_var,
                                   dpy_arr target_mean, dpy_arr target_var){
               return from_mat<double>(obj->forward(eps_mu, eps_sig,
                                                    to_vec<double>(old_mean), to_vec<double>(old_var),
                                                    to_vec<double>(target_mean), to_vec<double>(target_var)));},
           py::arg("eps_mu"), py::arg("eps_sig"), py::arg("old_mean"),
           py::arg("old_var"), py::arg("target_mean"), py::arg("target_var"));

    mp.def("backward", [](SplitDiagMoreProjection* obj, dpy_arr dl_dmu_projected, dpy_arr dl_dvar_projected){
               vec dl_dmu_target;
               vec dl_dvar_target;
               std::tie(dl_dmu_target, dl_dvar_target) = obj->backward(to_vec<double>(dl_dmu_projected),
                                                                         to_vec<double>(dl_dvar_projected));
               return std::make_tuple(from_mat<double>(dl_dmu_target), from_mat<double>(dl_dvar_target));},
           py::arg("dl_dmu_projected"), py::arg("dl_var_projected"));

    /* ------------------------------------------------------------------------------
    BATCHED PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<BatchedProjection> bp(p, "BatchedProjection");
    bp.def(py::init([](uword batch_size, uword dim, bool eec, bool constrain_entropy, int max_eval){
        return new BatchedProjection(batch_size, dim, eec, constrain_entropy, max_eval);}),
        py::arg("batchsize"), py::arg("dim"), py::arg("eec"), py::arg("constrain_entropy"),
        py::arg("max_eval") = 100);

    bp.def("forward", [](BatchedProjection* obj, dpy_arr epss, dpy_arr betas,
                           dpy_arr old_means, dpy_arr old_covars, dpy_arr target_means, dpy_arr target_covars){
        mat means;
        cube covs;
        try {
            std::tie(means, covs) = obj->forward(
                    to_vec<double>(epss), to_vec<double>(betas),
                    to_mat<double>(old_means), to_cube<double>(old_covars),
                    to_mat<double>(target_means), to_cube<double>(target_covars));
        } catch (std::invalid_argument &e) {
            PyErr_SetString(PyExc_AssertionError, e.what());
        }
        return std::make_tuple(from_mat(means), from_cube(covs));
        },
           py::arg("epss"), py::arg("beta"), py::arg("old_mean"),
           py::arg("old_covar"), py::arg("target_mean"), py::arg("target_covar")
    );

    bp.def("backward", [](BatchedProjection* obj, dpy_arr d_means, dpy_arr d_covs){
        mat d_means_target;
        cube d_covs_target;
        std::tie(d_means_target, d_covs_target) = obj->backward(to_mat<double>(d_means), to_cube<double>(d_covs));
        return std::make_tuple(from_mat(d_means_target), from_cube(d_covs_target));
        },
           py::arg("d_means"), py::arg("d_covs"));


    /* ------------------------------------------------------------------------------
    BATCHED DIAG COVAR ONLY PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<BatchedDiagCovOnlyProjection> bdcop(p, "BatchedDiagCovOnlyProjection");
    bdcop.def(py::init([](uword batch_size, uword dim, int max_eval){
        return new BatchedDiagCovOnlyProjection(batch_size, dim, max_eval);}),
           py::arg("batchsize"), py::arg("dim"), py::arg("max_eval") = 100);

    bdcop.def("forward", [](BatchedDiagCovOnlyProjection* obj, dpy_arr epss, dpy_arr old_vars,
            dpy_arr target_vars){
           try {
                   mat vars = obj->forward(to_vec<double>(epss), to_mat<double>(old_vars), to_mat<double>(target_vars));
                   return from_mat_enforce_mat<double>(vars);
               } catch (std::invalid_argument &e) {
                   PyErr_SetString(PyExc_AssertionError, e.what());
               }
           },
           py::arg("epss"),py::arg("old_var"), py::arg("target_var")
    );

    bdcop.def("backward", [](BatchedDiagCovOnlyProjection* obj, dpy_arr d_vars){
               mat d_vars_d_target = obj->backward(to_mat<double>(d_vars));
               return from_mat_enforce_mat<double>(d_vars_d_target);}, py::arg("d_vars"));

    /* ------------------------------------------------------------------------------
    BATCHED COVAR ONLY PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<BatchedCovOnlyProjection> bcop(p, "BatchedCovOnlyProjection");
    bcop.def(py::init([](uword batch_size, uword dim, int max_eval){
                  return new BatchedCovOnlyProjection(batch_size, dim, max_eval);}),
              py::arg("batchsize"), py::arg("dim"), py::arg("max_eval") = 100);

    bcop.def("forward", [](BatchedCovOnlyProjection* obj, dpy_arr epss, dpy_arr old_chols,
            dpy_arr target_chols, dpy_arr target_vars){
                  try {
                      cube covars = obj->forward(to_vec<double>(epss), to_cube<double>(old_chols),
                                            to_cube<double>(target_chols), to_cube<double>(target_vars));
                      return from_cube<double>(covars);
                  } catch (std::invalid_argument &e) {
                      PyErr_SetString(PyExc_AssertionError, e.what());
                  }
              },
              py::arg("epss"),py::arg("old_chols"),py::arg("target_chols"), py::arg("target_covar")
    );

    bcop.def("backward", [](BatchedCovOnlyProjection* obj, dpy_arr d_vars){
        cube d_covars_d_target = obj->backward(to_cube<double>(d_vars));
        return from_cube<double>(d_covars_d_target);}, py::arg("d_vars"));

    /* ------------------------------------------------------------------------------
    BATCHED SPLIT DIAG COVAR PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<BatchedSplitDiagMoreProjection> bsdcmp(p, "BatchedSplitDiagMoreProjection");
    bsdcmp.def(py::init([](uword batch_size, uword dim, int max_eval){
               return new BatchedSplitDiagMoreProjection(batch_size, dim, max_eval);}),
           py::arg("batchsize"), py::arg("dim"), py::arg("max_eval") = 100);

    bsdcmp.def("forward", [](BatchedSplitDiagMoreProjection* obj, dpy_arr epss, dpy_arr betas,
                         dpy_arr old_means, dpy_arr old_vars, dpy_arr target_means, dpy_arr target_vars){
               mat means;
               mat vars;
               try {
                   std::tie(means, vars) = obj->forward(
                           to_vec<double>(epss), to_vec<double>(betas),
                           to_mat<double>(old_means), to_mat<double>(old_vars),
                           to_mat<double>(target_means), to_mat<double>(target_vars));
               } catch (std::invalid_argument &e) {
                   PyErr_SetString(PyExc_AssertionError, e.what());
               }
               return std::make_tuple(from_mat_enforce_mat(means), from_mat_enforce_mat(vars));
           },
           py::arg("epss"), py::arg("beta"), py::arg("old_mean"),
           py::arg("old_vars"), py::arg("target_mean"), py::arg("target_vars")
    );


    bsdcmp.def("backward", [](BatchedSplitDiagMoreProjection* obj, dpy_arr d_means, dpy_arr d_vars){
               mat d_means_target;
               mat d_vars_target;
               std::tie(d_means_target, d_vars_target) = obj->backward(to_mat<double>(d_means), to_mat<double>(d_vars));
               return std::make_tuple(from_mat_enforce_mat(d_means_target), from_mat_enforce_mat(d_vars_target));
           },
           py::arg("d_means"), py::arg("d_vars"));

}