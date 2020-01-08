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
  selected_index <- which(mm > cutoff)
  return(selected_index)
}


### calculate fdp and power
fdp_power <- function(selected_index){
  fdp <- (length(selected_index) - length(intersect(selected_index, signal_index)))/max(length(selected_index), 1)
  power <- length(intersect(selected_index, signal_index))/length(signal_index)
  return(list(fdp = fdp, power = power))
}


### data-splitting methods (DS and MDS)
DS <- function(X, y, num_split, q){
  n <- dim(X)[1]; p <- dim(X)[2]
  inclusion_rate <- matrix(0, nrow = num_split, ncol = p)
  fdp <- rep(0, num_split)
  power <- rep(0, num_split)
  num_select <- rep(0, num_split)
  for(iter in 1:num_split){
    ### randomly split the data
    sample_index1 <- sample(x = c(1:n), size = 0.5 * n, replace = F)
    sample_index2 <- setdiff(c(1:n), sample_index1)
    ### get the penalty lambda for Lasso
    cvfit <- cv.glmnet(X[sample_index1, ], y[sample_index1], type.measure = "mse", nfolds = 10)
    lambda <- cvfit$lambda.1se
    ### run Lasso on the first half of the data
    beta1 <- as.vector(glmnet(X[sample_index1, ], y[sample_index1], family = "gaussian", alpha = 1, lambda = lambda)$beta)
    nonzero_index <- which(beta1 != 0)
    ### run OLS on the second half of the data, restricted on the selected features
    beta2 <- rep(0, p)
    beta2[nonzero_index] <- as.vector(lm(y[sample_index2] ~ X[sample_index2, nonzero_index] - 1)$coeff)
    
    ### calculate the mirror statistics
    M <- abs(beta1 + beta2) - abs(beta1 - beta2)
    selected_index <- analys(M, abs(M), q)
    ### number of selected variables
    num_select[iter] <- length(selected_index)
    inclusion_rate[iter, selected_index] <- 1/num_select[iter]
    ### calculate fdp and power
    fdp_power_result <- fdp_power(selected_index)
    fdp[iter] <- fdp_power_result$fdp
    power[iter] <- fdp_power_result$power
  }
  
  ### single data-splitting (DS) result
  DS_fdp <- fdp[1]
  DS_power <- power[1]
  
  ### multiple data-splitting (MDS) result
  inclusion_rate <- apply(inclusion_rate, 2, mean)
  ### rank the features by the empirical inclusion rate
  feature_rank <- order(inclusion_rate)
  feature_rank <- setdiff(feature_rank, which(inclusion_rate == 0))
  null_feature <- numeric()
  ### backtracking 
  for(feature_index in 1:length(feature_rank)){
    if(sum(inclusion_rate[feature_rank[1:feature_index]]) > q){
      break
    }else{
      null_feature <- c(null_feature, feature_rank[feature_index])
    }
  }
  selected_index <- setdiff(feature_rank, null_feature)
  ### calculate fdp and power
  fdp_power_result <- fdp_power(selected_index)
  MDS_fdp <- fdp_power_result$fdp
  MDS_power <- fdp_power_result$power
  
  return(list(DS_fdp = DS_fdp, DS_power = DS_power, MDS_fdp = MDS_fdp, MDS_power = MDS_power))
}

### model-X knockoff filter (Candes et al. 2018)
M_knockoff <- function(X, y, q){
  knockoff_result <- knockoff.filter(X, y, fdr = q, offset = 0)
  selected_index <- knockoff_result$selected
  
  ### calculate fdp and power
  fdp_power_result <- fdp_power(selected_index)
  knockoff_fdp <- fdp_power_result$fdp
  knockoff_power <- fdp_power_result$power
  return(list(fdp = knockoff_fdp, power = knockoff_power))
}

### fixed-design knockoff filter based on data splitting and data recycling (Barbers and Candes 2019)
F_knockoff <- function(X, y, q){
  ### randomly split the data
  sample_index1 <- sample(x = c(1:n), size = 0.5 * n, replace = F)
  sample_index2 <- setdiff(c(1:n), sample_index1)
  ### feature screening via Lasso
  fit <- glmnet(X[sample_index1, ], y[sample_index1], family = "gaussian", alpha = 1)
  beta1_mat <- fit$beta
  lambda <- fit$lambda
  final_lambda <- numeric(p)
  for(i in 1:p){
    final_lambda[i] <- lambda[max(which(beta1_mat[i,] == 0))]
  }
  sorted_lambda <- sort(final_lambda, decreasing = T, index.return = T)
  nonzero_index <- sorted_lambda$ix[1:trunc(n/4)]
  ### create knockoff features
  X2_knockoff <- create.fixed(X[sample_index2, nonzero_index])$X
  X1_knockoff <- X[sample_index1, nonzero_index]
  X_knockoff <- rbind(X1_knockoff, X2_knockoff)
  X1 <- X[sample_index1, nonzero_index]
  X2 <- X[sample_index2, nonzero_index]
  X <- rbind(X1, X2)
  y1 <- y[sample_index1]
  y2 <- y[sample_index2]
  y <- c(y1, y2)
  
  M <- stat.glmnet_coefdiff(X, X_tilde, y)
  cutoff <- knockoff.threshold(M, fdr = q, offset = 0)
  selected_index <- nonzero_index[sort(which(M >= cutoff))]
  ### calculate fdp and power
  fdp_power_result <- fdp_power(selected_index)
  fdp <- fdp_power_result$fdp
  power <- fdp_power_result$power
  return(list(fdp = fdp, power = power))
}


BH_BY_single <- function(X, y, q){
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
  harmonic_sum <- sum(1/(1:length(nonzero_index)))
  sorted_pvalues <- sort(pvalues, decreasing = F, index.return = T)
  
  BH_cutoff <- max(which(sorted_pvalues$x <= (1:length(nonzero_index))*q/length(nonzero_index)))
  BY_cutoff <- max(which(sorted_pvalues$x <= (1:length(nonzero_index))*q/(length(nonzero_index)*harmonic_sum)))
  BH_selected_index <- nonzero_index[sorted_pvalues$ix[1:BH_cutoff]]
  BY_selected_index <- nonzero_index[sorted_pvalues$ix[1:BY_cutoff]]
  
  ### calculate fdp and power
  BH_fdp_power_result <- fdp_power(BH_selected_index)
  BH_fdp <- BH_fdp_power_result$fdp
  BH_power <- BH_fdp_power_result$power
  BY_fdp_power_result <- fdp_power(BY_selected_index)
  BY_fdp <- BY_fdp_power_result$fdp
  BY_power <- BY_fdp_power_result$power
  
  return(list(BH_fdp = BH_fdp, BH_power = BH_power, BY_fdp = BY_fdp, BY_power = BY_power))
}

BH_BY_multiple <- function(X, y, q, num_split){
  multi_fit <- multi.split(X, y, B = num_split)
  pvalues <- multi_fit$pval.corr
  sorted_pvalues <- sort(pvalues, decreasing = F, index.return = T)
  selected_index <- which(sorted_pvalues$x < 1)
  harmonic_sum <- sum(1/(1:length(selected_index)))
  BH_cutoff <- max(which(sorted_pvalues$x[selected_index] <= (1:length(selected_index))*q))
  BY_cutoff <- max(which(sorted_pvalues$x[selected_index] <= (1:length(selected_index))*q/harmonic_sum))
  BH_selected_index <- sorted_pvalues$ix[1:BH_cutoff]
  BY_selected_index <- sorted_pvalues$ix[1:BY_cutoff]
  
  ### calculate fdp and power
  BH_fdp_power_result <- fdp_power(BH_selected_index)
  BH_fdp <- BH_fdp_power_result$fdp
  BH_power <- BH_fdp_power_result$power
  BY_fdp_power_result <- fdp_power(BY_selected_index)
  BY_fdp <- BY_fdp_power_result$fdp
  BY_power <- BY_fdp_power_result$power
  
  return(list(BH_fdp = BH_fdp, BH_power = BH_power, BY_fdp = BY_fdp, BY_power = BY_power))
}

### test out different methods
DS_result <- DS(X, y, num_split, q)
M_knockoff_result <- M_knockoff(X, y, q)
F_knockoff_result <- F_knockoff(X, y, q)
BH_BY_single_result <- BH_BY_single(X, y, q)
BH_BY_multiple_result <- BH_BY_multiple(X, y, q, num_split)


