##### Gaussian graphical model
rm(list = ls())
library(mvtnorm)
library(glmnet)
library(SILGGM)
library(ppcor)

##### banded graphs
### replicate index
replicate <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
set.seed(replicate)
p <- 100
rho <- 8
a <- -0.6
n <- as.numeric(Sys.getenv("att"))
c <- 1.5
q <- 0.2
num_split <- 50
precision <- matrix(0, nrow = p, ncol = p)
edges_set <- matrix(0, nrow = p, ncol = p)
for(i in 1:p){
  for(j in 1:p){
    if(i == j){
      precision[i, j] <- 1
    }
    if(i != j & abs(i - j) <= rho){
      precision[i, j] <- sign(a)*abs(a)^(abs(i - j)/c)
      edges_set[i, j] <- 1
    }
  }
}
min_eigen <- min(eigen(precision)$values)
if(min_eigen < 0){diag(precision) <- diag(precision) + abs(min_eigen) + 0.005}

##### generate samples
data <- rmvnorm(n, mean = rep(0, p), sigma = solve(precision))

#### nonzero_index_selection
analys = function(mm, ww, q){
  t_set = max(ww)
  for(t in ww){
    ps = length(mm[mm>=t])
    ng = length(na.omit(mm[mm<=-t]))
    rto = (ng)/max(ps, 1)
    if(rto<=q){
      t_set = c(t_set, t)
    }
  }
  thre = min(t_set)
  nz_est = which(mm>thre)
  nz_est
}

##### Together lasso selection
DS_Together_single = function(data, q){
  beta1mat = matrix(0, nrow = p, ncol = p)
  beta2mat = matrix(0, nrow = p, ncol = p)
  sample_index1 <- sample(x = c(1:n), size = 0.5 * n, replace = F)
  sample_index2 <- setdiff(c(1:n), sample_index1)
  
  for(j in 1:p){
    ### response variable and design matrix
    y <- data[, j]
    X <- data[, -j]
    
    ### estimate the regularization parameter
    cvfit <- cv.glmnet(X[sample_index1, ], y[sample_index1], type.measure = "mse", nfolds = 10)
    lambda <- cvfit$lambda.1se
    beta1 <- as.vector(glmnet(X[sample_index1, ], y[sample_index1], family = "gaussian", alpha = 1, lambda = lambda)$beta)
    nonzero_index = which(beta1!=0)
    if(length(nonzero_index) != 0){
      beta2 <- rep(0, p - 1)
      fit = lm(y[sample_index2] ~ X[sample_index2, nonzero_index] - 1)
      beta2[nonzero_index] <- as.vector(fit$coeff)
      beta1mat[j, -j] = beta1/sum(fit$residuals^2)*(0.5 * n - length(nonzero_index))
      beta2mat[j, -j] = beta2/sum(fit$residuals^2)*(0.5 * n - length(nonzero_index))
    }
  }
  beta1mat = (beta1mat + t(beta1mat))/2
  beta2mat = (beta2mat + t(beta2mat))/2
  Mmat = abs(beta1mat + beta2mat) - abs(beta1mat - beta2mat)
  test_stat = Mmat[upper.tri(Mmat)]
  selection <- matrix(0, nrow = p, ncol = p)
  nz_est = analys(test_stat, abs(test_stat), q)
  selection[upper.tri(selection)][nz_est] = 1
  
  selection
}

DS_Together = function(data, q, num_split){
  selection_list = lapply(1:num_split, function(o) DS_Together_single(data, q))
  inclusion_mat = matrix(0, nrow = p, ncol = p)
  for(i in 1:num_split){
    inclusion_mat = inclusion_mat+selection_list[[i]]/sum(selection_list[[i]])
  }
  inclusion_rate = inclusion_mat[upper.tri(inclusion_mat)]/num_split
  feature_rank = order(inclusion_rate)
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
  nz_est = selected_index
  
  selection = matrix(0, nrow = p, ncol = p)
  selection[upper.tri(selection)][nz_est] = 1
  false_discoveries <- 0
  num_selected_edge <- 0
  for(i in 1:(p - 1)){
    for(j in (i + 1):p){
      if(selection[i, j] == 1){
        num_selected_edge <- num_selected_edge + 1
        if(edges_set[i, j] == 0){
          false_discoveries <- false_discoveries + 1
        }
      }
    }
  }
  DS_fdr_multiple <- false_discoveries/num_selected_edge
  DS_power_multiple <- (num_selected_edge - false_discoveries)/sum(edges_set)*2
  
  selection = selection_list[[1]]
  false_discoveries <- 0
  num_selected_edge <- 0
  for(i in 1:(p - 1)){
    for(j in (i + 1):p){
      if(selection[i, j] == 1){
        num_selected_edge <- num_selected_edge + 1
        if(edges_set[i, j] == 0){
          false_discoveries <- false_discoveries + 1
        }
      }
    }
  }
  DS_fdr_single <- false_discoveries/num_selected_edge
  DS_power_single <- (num_selected_edge - false_discoveries)/sum(edges_set)*2
  list(multiple_split_together_fdr = DS_fdr_multiple, multiple_split_together_power = DS_power_multiple, 
       single_split_together_fdr = DS_fdr_single, single_split_together_power = DS_power_single)
}

## Nodewise data splitting procedure
DS_NodeWise = function(data, q, num_split){
  single_selection <- matrix(0, nrow = p, ncol = p)
  multiple_selection <- matrix(0, nrow = p, ncol = p)
  for(j in 1:p){
    ### response variable and design matrix
    y <- data[, j]
    X <- data[, -j]
    
    inclusion_rate_multiple <- matrix(0, nrow = num_split, ncol = p - 1)
    num_select <- rep(0, num_split)
    
    for(iter in 1:num_split){
      ### randomly split the data and estimate coefficient
      sample_index1 <- sample(x = c(1:n), size = 0.5 * n, replace = F)
      sample_index2 <- setdiff(c(1:n), sample_index1)
      ### estimate the regularization parameter
      cvfit <- cv.glmnet(X[sample_index1, ], y[sample_index1], type.measure = "mse", nfolds = 10)
      lambda <- cvfit$lambda.1se
      beta1 <- as.vector(glmnet(X[sample_index1, ], y[sample_index1], family = "gaussian", alpha = 1, lambda = lambda)$beta)
      nonzero_index = which(beta1!=0)
      if(length(nonzero_index)!=0){
        beta2 <- rep(0, p - 1)
        fit = lm(y[sample_index2] ~ X[sample_index2, nonzero_index] - 1)
        beta2[nonzero_index] <- as.vector(fit$coeff)
      }
      
      ### calculate the test statistics
      M <- abs(beta1 + beta2) - abs(beta1 - beta2)
      current_selected_index <- analys(M, abs(M), q/2)
      ### number of selected variables
      num_select[iter] <- length(current_selected_index)
      inclusion_rate_multiple[iter, current_selected_index] <- 1/num_select[iter]
    }
    
    ### single splitting result
    single_selection[j, -j] <- ifelse(inclusion_rate_multiple[1, ] > 0, 1, 0)
    ### multiple splitting result
    inclusion_rate <- apply(inclusion_rate_multiple, 2, mean)
    feature_rank <- order(inclusion_rate)
    feature_rank <- setdiff(feature_rank, which(inclusion_rate == 0))
    null_variable <- numeric()
    for(feature_index in 1:length(feature_rank)){
      if(sum(inclusion_rate[feature_rank[1:feature_index]]) > q/2){
        break
      }else{
        null_variable <- c(null_variable, feature_rank[feature_index])
      }
    }
    selected_index = rep(0, p - 1)
    selected_index[setdiff(feature_rank, null_variable)] = 1
    multiple_selection[j, -j] <- selected_index
  }
  
  ##### single split fdr and power
  false_discoveries <- 0
  num_selected_edge <- 0
  for(i in 1:(p - 1)){
    for(j in (i + 1):p){
      if(single_selection[i, j] == 1 | single_selection[j, i] == 1){
        num_selected_edge <- num_selected_edge + 1
        if(edges_set[i, j] == 0){
          false_discoveries <- false_discoveries + 1
        }
      }
    }
  }
  single_split_fdr <- false_discoveries/num_selected_edge
  single_split_power <- (num_selected_edge - false_discoveries)/sum(edges_set)*2
  
  
  ##### multiple split fdr and power
  false_discoveries <- 0
  num_selected_edge <- 0
  for(i in 1:(p - 1)){
    for(j in (i + 1):p){
      if(multiple_selection[i, j] == 1 | multiple_selection[j, i] == 1){
        num_selected_edge <- num_selected_edge + 1
        if(edges_set[i, j] == 0){
          false_discoveries <- false_discoveries + 1
        }
      }
    }
  }
  multiple_split_fdr <- false_discoveries/num_selected_edge
  multiple_split_power <- (num_selected_edge - false_discoveries)/sum(edges_set)*2
  
  list(single_split_nodewise_fdr = single_split_fdr, single_split_nodewise_power = single_split_power,
       multiple_split_nodewise_fdr = multiple_split_fdr, multiple_split_nodewise_power = multiple_split_power)
}


##### comparison to Liu et al (2013)
Liu = function(x, precision, q){
  fit1 = SILGGM(x, method = 'GFC_L',  true_graph = precision, alpha = q)
  fit2 = SILGGM(x, method = 'GFC_SL', true_graph = precision, alpha = q)
  list(fdr = c(fit1$FDR, fit2$FDR), power = c(fit1$power, fit2$power))
}



##### comparison to BH and BY
BH <- function(x, q){
  n = dim(x)[1]
  p = dim(x)[2]
  total = sum(1/(1:(p*(p-1)/2)))
  
  #### get pvalues
  pvalues = c(NULL)
  for(i in 2:p){
    for(j in 1:(i-1)){
      pvalues = suppressWarnings(c(pvalues, pcor.test(x[, i], x[, j], x[, -c(i,j)])$p.value))
    }
  }
  
  #### BH and BY methods
  sorted_pvalues = sort(pvalues, decreasing = F, index.return = T)
  BH_index = max(which(sorted_pvalues$x<=(1:(p*(p-1)/2))*q/(p*(p-1)/2)))
  BY_index = max(which(sorted_pvalues$x<=(1:(p*(p-1)/2))*q/(p*(p-1)/2*total)))
  nz_est_BH = sorted_pvalues$ix[1:BH_index]
  nz_est_BY = sorted_pvalues$ix[1:BY_index]
  
  ### calculate fdr and power
  nonzero = c(NULL)
  count = 0
  for(i in 2:p){
    for(j in 1:(i-1)){
      count <- count + 1
      if(precision[j,i] != 0){
        nonzero = c(nonzero, count)
      }
    }
  }
  BH_inc = intersect(nonzero, nz_est_BH)
  BH_td = length(BH_inc)
  BH_fdr = (length(nz_est_BH) - BH_td)/max(length(nz_est_BH), 1)
  BH_power = BH_td/length(nonzero)
  
  BY_inc = intersect(nonzero, nz_est_BY)
  BY_td = length(BY_inc)
  BY_fdr = (length(nz_est_BY) - BY_td)/max(length(nz_est_BY), 1)
  BY_power = BY_td/length(nonzero)  
  
  return(list(BH_fdr = BH_fdr, BH_power = BH_power, BY_fdr = BY_fdr, BY_power = BY_power))
}


BH_result <- BH(data, q)
DS_nodewise_result <- DS_NodeWise(data, q, num_split)
DS_together_result <- DS_Together(data, q, num_split)
GFC_result <- Liu(data, precision, q)



##### save data
data_save <- list(DS_nodewise_single_fdr = DS_nodewise_result$single_split_nodewise_fdr, 
                  DS_nodewise_single_power = DS_nodewise_result$single_split_nodewise_power, 
                  DS_nodewise_multiple_fdr = DS_nodewise_result$multiple_split_nodewise_fdr, 
                  DS_nodewise_multiple_power = DS_nodewise_result$multiple_split_nodewise_power, 
                  
                  DS_together_single_fdr = DS_together_result$single_split_together_fdr, 
                  DS_together_single_power = DS_together_result$single_split_together_power,
                  DS_together_multiple_fdr = DS_together_result$multiple_split_together_fdr, 
                  DS_together_multiple_power = DS_together_result$multiple_split_together_power,
                  
                  GFC_L_fdr = GFC_result$fdr[1], GFC_L_power = GFC_result$power[1],
                  GFC_SL_fdr = GFC_result$fdr[2], GFC_SL_power = GFC_result$power[2],
                  BH_fdr = BH_result$BH_fdr, BH_power = BH_result$BH_power,
                  BY_fdr = BH_result$BY_fdr, BY_power = BH_result$BY_power)

save(data_save, file = paste("/n/home09/cdai/FDR/result/GraphicalModel/Banded/vary_n/n_", n, "_replicate_", replicate, ".RData", sep = ""))
