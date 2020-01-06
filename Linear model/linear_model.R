### FDR control for high-dimensional linear model
rm(list = ls())
library(MASS)
library(glmnet)
library(knockoff)
library(mvtnorm)
library(hdi)

### algorithmic settings
n <- 500
p <- 500
p0 <- 50
q <- 0.1
rho <- 0.5 ### rho is the pairwise correlation between features
delta <- 4 ### delta is the signal strength

### pairwise constant correlation
covariance <- rep(1, p)%*%t(rep(rho, p))
diag(covariance) <- rep(1, p)
X <- mvrnorm(n, mu = rep(0, p), Sigma = covariance)

### generate the true regression coefficient
beta_star <- rep(0, p)
signal_index <- sample(c(1:p), size = p0, replace = F)
beta_star[signal_index] <- rnorm(p0, mean = 0, sd = delta*sqrt(log(p)/n))

### generate the response y
y <- X%*%beta_star + rnorm(n, mean = 0, sd = 1)
### number of multiple splits
num_split <- 50

### select the relevant features using mirror statistics
analys <- function(mm, ww, q){
  ### mm: mirror statistics
  ### ww: absolute value of mirror statistics
  ### q:  FDR control level
  cutoff_set <- max(ww)
  for(t in ww){
    ps <- length(mm[mm > t])
    ng <- length(na.omit(mm[mm < -t]))
    rto <- (ng)/max(ps, 1)
    if(rto <= q){
      cutoff_set <- c(cutoff_set, t)
    }
  }
  cutoff <- min(cutoff_set)
  selected_fetaure_index <- which(mm > cutoff)
  return(selected_feature_index)
}


DS <- function(X, y, num_split, q){
  n = dim(X)[1]; p = dim(X)[2]
  inclusion_rate_multiple <- matrix(0, nrow = num_split, ncol = p)
  fdr_multiple <- rep(0, num_split)
  power_multiple <- rep(0, num_split)
  num_select <- rep(0, num_split)
  for(iter in 1:num_split){
    ### randomly split the data and estimate coefficient
    sample_index1 <- sample(x = c(1:n), size = 0.5 * n, replace = F)
    sample_index2 <- setdiff(c(1:n), sample_index1)
    ### get penalty lambda
    cvfit <- cv.glmnet(X[sample_index1, ], y[sample_index1], type.measure = "mse", nfolds = 10)
    lambda <- cvfit$lambda.1se
    beta1 <- as.vector(glmnet(X[sample_index1, ], y[sample_index1], family = "gaussian", alpha = 1, lambda = lambda)$beta)
    nonzero_index <- which(beta1 != 0)
    beta2 <- rep(0, p)
    beta2[nonzero_index] <- as.vector(lm(y[sample_index2] ~ X[sample_index2, nonzero_index] - 1)$coeff)
    
    ### calculate the test statistics
    M <- abs(beta1 + beta2) - abs(beta1 - beta2)
    current_selected_index <- analys(M, abs(M), q)
    ### number of selected variables
    num_select[iter] <- length(current_selected_index)
    inclusion_rate_multiple[iter, current_selected_index] <- 1/num_select[iter]
    ### false discovery rate
    fdr_multiple[iter] <- (length(current_selected_index) - length(intersect(current_selected_index, signal_index)))/num_select[iter]
    ### power
    power_multiple[iter] <- length(intersect(current_selected_index, signal_index))/p0
    
  }
  ### single splitting result
  single_split_fdr <- fdr_multiple[1]
  single_split_power <- power_multiple[1]
  ### multiple splitting result
  inclusion_rate <- apply(inclusion_rate_multiple, 2, mean)
  feature_rank <- order(inclusion_rate)
  feature_rank <- setdiff(feature_rank, which(inclusion_rate == 0))
  null_variable <- numeric()
  for(feature_index in 1:length(feature_rank)){
    if(sum(inclusion_rate[feature_rank[1:feature_index]]) > q){
      break
    }else{
      null_variable <- c(null_variable, feature_rank[feature_index])
    }
  }
  ### conservative one
  selected_index <- setdiff(feature_rank, null_variable)
  multiple_splitting_fdr <- (length(selected_index) - length(intersect(selected_index, signal_index)))/length(selected_index)
  multiple_splitting_power <- length(intersect(selected_index, signal_index))/p0
  
  list(SDS_fdr = single_split_fdr, SDS_power = single_split_power, MDS_fdr = multiple_splitting_fdr, MDS_power = multiple_splitting_power)
}

## Knockoff Result
KN = function(X, y, q){
  knockoff_result <- knockoff.filter(X, y, fdr = q, offset = 0)
  selected_index <- knockoff_result$selected
  ### power
  knockoff_fdr <- (length(selected_index) - length(intersect(selected_index, signal_index)))/length(selected_index)
  knockoff_power <- length(intersect(selected_index, signal_index))/p0
  list(fdr = knockoff_fdr, power = knockoff_power)
}

BH_and_BY_single = function(X, y, q){
  #### get penalty lambda
  cvfit <- cv.glmnet(X, y, type.measure = "mse", nfolds = 10)
  lambda <- cvfit$lambda.1se
  lambda <- lambda/sqrt(2)
  
  #### get pvalues
  sample_index1 <- sample(x = c(1:n), size = 0.5 * n, replace = F)
  sample_index2 <- setdiff(c(1:n), sample_index1)
  beta1 <- as.vector(glmnet(X[sample_index1, ], y[sample_index1], family = "gaussian", alpha = 1, lambda = lambda)$beta)
  nonzero_index = which(beta1!=0)
  fit <- lm(y[sample_index2] ~ X[sample_index2, nonzero_index] - 1)
  pvalues <- summary(fit)$coefficients[, 4]
  
  #### BH and BY procedure
  total = sum(1/(1:length(nonzero_index)))
  sorted_pvalues = sort(pvalues, decreasing = F, index.return = T)
  
  BH_index = max(which(sorted_pvalues$x<=(1:length(nonzero_index))*q/length(nonzero_index)))
  BY_index = max(which(sorted_pvalues$x<=(1:length(nonzero_index))*q/(length(nonzero_index)*total)))
  nz_est_BH = nonzero_index[sorted_pvalues$ix[1:BH_index]]
  nz_est_BY = nonzero_index[sorted_pvalues$ix[1:BY_index]]
  
  ### calculate fdr and power
  BH_inc = intersect(signal_index, nz_est_BH)
  BH_td = length(BH_inc)
  BH_fdr = (length(nz_est_BH) - BH_td)/max(length(nz_est_BH), 1)
  BH_power = BH_td/length(signal_index)
  
  BY_inc = intersect(signal_index, nz_est_BY)
  BY_td = length(BY_inc)
  BY_fdr = (length(nz_est_BY) - BY_td)/max(length(nz_est_BY), 1)
  BY_power = BY_td/length(signal_index)  
  
  return(list(BH_fdr = BH_fdr, BH_power = BH_power, BY_fdr = BY_fdr, BY_power = BY_power))
}

BH_and_BY_multiple = function(X, y, q, num_split){
  multi_fit = multi.split(X, y, B = num_split)
  pvalues = multi_fit$pval.corr
  sorted_pvalues = sort(pvalues, decreasing = F, index.return = T)
  first_selection = which(sorted_pvalues$x<1)
  total = sum(1/(1:length(first_selection)))
  BH_index = max(which(sorted_pvalues$x[first_selection]<=(1:length(first_selection))*q))
  BY_index = max(which(sorted_pvalues$x[first_selection]<=(1:length(first_selection))*q/total))
  nz_est_BH_multiple = sorted_pvalues$ix[1:BH_index]
  nz_est_BY_multiple = sorted_pvalues$ix[1:BY_index]
  
  BH_inc = intersect(signal_index, nz_est_BH_multiple)
  BH_td = length(BH_inc)
  BH_fdr = (length(nz_est_BH_multiple) - BH_td)/max(length(nz_est_BH_multiple), 1)
  BH_power = BH_td/length(signal_index)
  
  BY_inc = intersect(signal_index, nz_est_BY_multiple)
  BY_td = length(BY_inc)
  BY_fdr = (length(nz_est_BY_multiple) - BY_td)/max(length(nz_est_BY_multiple), 1)
  BY_power = BY_td/length(signal_index)
  return(list(BH_fdr = BH_fdr, BH_power = BH_power, BY_fdr = BY_fdr, BY_power = BY_power))
}

KN_hdi = function(X, y, q){
  sample_index1 <- sample(x = c(1:n), size = 0.5 * n, replace = F)
  sample_index2 <- setdiff(c(1:n), sample_index1)
  fit = glmnet(X[sample_index1, ], y[sample_index1], family = "gaussian", alpha = 1)
  beta1_mat = fit$beta
  lambda = fit$lambda
  final_lambda = numeric(p)
  for(i in 1:p){
    final_lambda[i] = lambda[max(which(beta1_mat[i,]==0))]
  }
  sorted_lambda = sort(final_lambda, decreasing = T, index.return = T)
  nonzero_index1 = sorted_lambda$ix[1:trunc(n/4)]
  X2_tilde = create.fixed(X[sample_index2, nonzero_index1])$X
  X1_tilde = X[sample_index1,nonzero_index1]
  X_tilde = rbind(X1_tilde, X2_tilde)
  X1 = X[sample_index1, nonzero_index1]
  X2 = X[sample_index2, nonzero_index1]
  X = rbind(X1, X2)
  y1 = y[sample_index1]
  y2 = y[sample_index2]
  y = c(y1,y2)
  
  M <- stat.glmnet_coefdiff(X, X_tilde, y)
  t = knockoff.threshold(M, fdr = q, offset = 0)
  selected = sort(which(M >= t))
  nz_est = nonzero_index1[selected]
  inc = intersect(signal_index, nz_est)
  td = length(inc)
  fdp = (length(nz_est)-td)/max(length(nz_est),1)
  power = td/length(signal_index)
  list(fdp = fdp, power = power)
}


###
DS_result <- DS(X,y, num_split, q)
knockoff_result <- KN(X, y, q)
BH_result <- BH_and_BY_single(X, y, q)
BH_multiple_result <- BH_and_BY_multiple(X, y, q, num_split)
knockoff_split_result <- KN_hdi(X, y, q)


### save data
data_save <- list(single_split_fdr = DS_result$SDS_fdr, single_split_power = DS_result$SDS_power,
                  multiple_splitting_fdr = DS_result$MDS_fdr,
                  multiple_splitting_power = DS_result$MDS_power,
                  knockoff_fdr = knockoff_result$fdr, knockoff_power = knockoff_result$power, 
                  BH_fdr = BH_result$BH_fdr, BH_power = BH_result$BH_power, 
                  BY_fdr = BH_result$BY_fdr, BY_power = BH_result$BY_power,
                  BH_multiple_fdr = BH_multiple_result$BH_fdr, BH_multiple_power = BH_multiple_result$BH_power,
                  BY_multiple_fdr = BH_multiple_result$BY_fdr, BY_multiple_power = BH_multiple_result$BY_power,
                  knockoff_split_fdr = knockoff_split_result$fdp,
                  knockoff_split_power = knockoff_split_result$power)
save(data_save, file = paste("/n/home09/cdai/FDR/result/LinearHighConstantCorrelation/delta_", delta, "_replicate_", replicate, ".RData", sep = ""))

