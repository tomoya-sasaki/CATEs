#' This function produces the predicted values of all 13 estimator considered in KLS
#'
#' @param y_t Vector of training outcome values
#' @param d_t Vector of training treament indicators
#' @param x_t Matrix of training covariates (N x p matrix)
#' @param x_v Matrix of validation covariates (N x p matrix)
#' @import grf glmnet caret
#'
#' @return Returns n x 13 matrix containing the IATEs of the validation sample
#'
#' @export

IATEs_old <- function(y_t,d_t,x_t,tau_t,x_v) {

  # Initialize matrix to store predicted IATEs of 13 estimators
  iates_mat = matrix(NA,n,13)
  colnames(iates_mat) =  c("RF Inf","RF CMR","RF MOM IPW","RF MOM DR","CF","CF LC",
                           "Lasso Inf","Lasso CMR","Lasso MOM IPW","Lasso MOM DR",
                           "Lasso MCM","Lasso MCM EA","Lasso RL")

  ### RF Forest based methods
  # Infeasible RF
  rf_inf = regression_forest(x_t, tau_t)
  iates_mat[,1] = predict(rf_inf, x_v)$predictions

  # CMR Random Forest
  rf0 = regression_forest(x_t[d_t==0,], y_t[d_t==0])
  rf1 = regression_forest(x_t[d_t==1,], y_t[d_t==1])
  iates_mat[,2] = predict(rf1, x_v)$predictions - predict(rf0, x_v)$predictions

  # Estimate RF nuisance parameters
  index = caret::createFolds(y_t, k = 2)

  np = nuisance_cf_grf(y_t,d_t,x_t,index)

  # MOMs with RF
  estimator_nm = list(mom_ipw_grf,mom_dr_grf)
  for (j in 1:2) {
    iates_mat[,j+2] = do.call(cf_dml1,list(estimator_nm[[j]],y_t,d_t,x_t,np,x_v,index))
  }

  # Causal Forest
  cf = causal_forest(x_t, y_t, d_t, Y.hat = rep(0,n), W.hat = rep(0,n))
  iates_mat[,5] = predict(cf,x_v)$predictions

  # Causal Forest with local centering
  cf_lc = causal_forest(x_t, y_t, d_t, Y.hat = np[,2], W.hat = np[,1])
  iates_mat[,6] = predict(cf_lc,x_v)$predictions

  ### Lasso based methods

  # Infeasible Lasso
  lasso_inf = cv.glmnet(x_t, tau_t)
  iates_mat[,7] = predict(lasso_inf, x_v)

  # CMR Lasso
  lasso0 = cv.glmnet(x_t[d_t==0,], y_t[d_t==0])
  lasso1 = cv.glmnet(x_t[d_t==1,], y_t[d_t==1])
  iates_mat[,8] = predict(lasso1, x_v) - predict(lasso0, x_v)

  # Estimate Lasso nuisance parameters with same index
  np = nuisance_cf_glmnet(y_t,d_t,x_t,index)

  # MOMs, MCMs and RL with Lasso
  estimator_nm = list(mom_ipw_glmnet,mom_dr_glmnet,
                      mcm_glmnet,mcm_ea_glmnet,rl_glmnet)
  for (j in 1:5) {
    iates_mat[,j+8] = do.call(cf_dml1,list(estimator_nm[[j]],y_t,d_t,x_t,np,x_v,index))
  }

  return(iates_mat)
}


#' This function produces the predicted values of all 11 estimators
#'
#' @param y_t Vector of training outcome values
#' @param d_t Vector of training treament indicators
#' @param x_t Matrix of training covariates (N x p matrix)
#' @param x_v Matrix of validation covariates (validation sample)
#'
#' @return Returns a list with length 11 which contains the IATEs of the validation sample

IATEs <- function(y_t,d_t,x_t,x_v) {

  # Initialize matrix to store predicted IATEs of 13 estimators
  # iates_mat = matrix(NA, nrow(x_v ), 13)
  iates_list <- list()
  N <- length(y_t)
  # names(iates_mat) =  c("RF Inf","RF CMR","RF MOM IPW","RF MOM DR","CF","CF LC",
  #                          "Lasso Inf","Lasso CMR","Lasso MOM IPW","Lasso MOM DR",
  #                          "Lasso MCM","Lasso MCM EA","Lasso RL")

  ### RF Forest based methods
  # Infeasible RF
  # rf_inf = regression_forest(x_t, tau_t)
  # iates_mat[,1] = predict(rf_inf, x_v)$predictions

  # CMR Random Forest
  rf0 <- grf::regression_forest(x_t[d_t==0,], y_t[d_t==0])
  rf1 <- grf::regression_forest(x_t[d_t==1,], y_t[d_t==1])
  iates_list[[1]] <- predict(rf1, x_v)$predictions - predict(rf0, x_v)$predictions

  # Estimate RF nuisance parameters
  index <- caret::createFolds(y_t, k = 2)

  np <- nuisance_cf_grf(y_t,d_t,x_t,index)

  # MOMs with RF
  estimator_nm <- list(mom_ipw_grf,mom_dr_grf)
  for (j in 1:2) {
    iates_list[[j+1]] <- do.call(cf_dml1,list(estimator_nm[[j]],y_t,d_t,x_t,np,x_v,index))
  }

  # Causal Forest
  cf <- grf::causal_forest(x_t, y_t, d_t, Y.hat = rep(0, N), W.hat = rep(0, N))
  iates_list[[4]] <- predict(cf,x_v)$predictions

  # Causal Forest with local centering
  cf_lc <- grf::causal_forest(x_t, y_t, d_t, Y.hat = np[,2], W.hat = np[,1])
  iates_list[[5]] <- predict(cf_lc,x_v)$predictions

  ### Lasso based methods

  # Infeasible Lasso
  # lasso_inf = cv.glmnet(x_t, tau_t)
  # iates_list[[7] = predict(lasso_inf, x_v)

  # CMR Lasso
  lasso0 <- glmnet::cv.glmnet(x_t[d_t==0,], y_t[d_t==0])
  lasso1 <- glmnet::cv.glmnet(x_t[d_t==1,], y_t[d_t==1])
  iates_list[[6]] <- predict(lasso1, x_v) - predict(lasso0, x_v)

  # Estimate Lasso nuisance parameters with same index
  np <- nuisance_cf_glmnet(y_t,d_t,x_t,index)

  # MOMs, MCMs and RL with Lasso
  estimator_nm <- list(mom_ipw_glmnet,mom_dr_glmnet,
                      mcm_glmnet,mcm_ea_glmnet,rl_glmnet)
  for (j in 1:5) {
    iates_list[[j+6]] <- do.call(cf_dml1,list(estimator_nm[[j]],y_t,d_t,x_t,np,x_v,index))
  }

  return(iates_list)
}

