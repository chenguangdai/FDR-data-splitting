### FDR control for Gaussian graphical model
rm(list = ls())
library(mvtnorm)
library(glmnet)
library(SILGGM)
library(ppcor)

### algorithmic setting
n <- 500    ### number of samples
p <- 100    ### number of vertexes
rho <- 8    ### sparsity of the neighborhood
a <- -0.6  
c <- 1.5
q <- 0.2
num_split <- 50
precision <- matrix(0, nrow = p, ncol = p)
edges_set <- matrix(0, nrow = p, ncol = p)

### banded graph
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

### block diagonal graph
# block_size <- 25
# num_blocks <- p/block_size
# for(iter in 1:num_blocks){
#   for(i in 1:block_size){
#     for(j in i:block_size){
#       row_index <- (iter - 1)*block_size + i
#       col_index <- (iter - 1)*block_size + j
#       if(row_index == col_index){
#         precision[row_index, col_index] <- 1
#       }else{
#         precision[row_index, col_index] <- runif(1, min = 0.4, max = 0.8) * sample(c(-1, 1), size = 1)
#         precision[col_index, row_index] <- precision[row_index, col_index]
#         edges_set[row_index, col_index] <- 1
#         edges_set[col_index, row_index] <- 1
#       }
#     }
#   }
# }

### the precision matrix should be positive definite
min_eigen <- min(eigen(precision)$values)
if(min_eigen < 0){diag(precision) <- diag(precision) + abs(min_eigen) + 0.005}

### generate samples
data <- rmvnorm(n, mean = rep(0, p), sigma = solve(precision))

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
    if(rto < q){
      cutoff_set <- c(cutoff_set, t)
    }
  }
  cutoff <- min(cutoff_set)
  selected_index <- which(mm > cutoff)
  return(selected_index)
}


### calculate fdp and power
fdp_power <- function(selected_edge){
  num_false_discoveries <- 0
  num_selected_edge <- 0
  for(i in 1:(p - 1)){
    for(j in (i + 1):p){
      if(selected_edge[i, j] == 1 | selected_edge[j, i] == 1){
        num_selected_edge <- num_selected_edge + 1
        if(edges_set[i, j] == 0){
          num_false_discoveries <- false_discoveries + 1
        }
      }
    }
  }
  fdp <- num_false_discoveries/num_selected_edge
  power <- (num_selected_edge - num_false_discoveries)/sum(edges_set)*2
  return(list(fdp = fdp, power = power))
}


### nodewise data-splitting procedure
DS <- function(data, q, num_split){
  DS_selected_edge <- matrix(0, nrow = p, ncol = p)
  MDS_selected_edge <- matrix(0, nrow = p, ncol = p)
  
  for(j in 1:p){
    ### nodewise selection
    ### response variable and design matrix
    y <- data[, j]
    X <- data[, -j]
    
    inclusion_rate <- matrix(0, nrow = num_split, ncol = p - 1)
    num_select <- rep(0, num_split)
    
    ### multiple data splits
    for(iter in 1:num_split){
      ### randomly split the data
      sample_index1 <- sample(x = c(1:n), size = 0.5 * n, replace = F)
      sample_index2 <- setdiff(c(1:n), sample_index1)
      ### get the penalty lambda for Lasso
      cvfit <- cv.glmnet(X[sample_index1, ], y[sample_index1], type.measure = "mse", nfolds = 10)
      lambda <- cvfit$lambda.1se
      ### run Lasso on the first half of the data
      beta1 <- as.vector(glmnet(X[sample_index1, ], y[sample_index1], family = "gaussian", alpha = 1, lambda = lambda)$beta)
      nonzero_index = which(beta1!=0)
      ### run OLS on the second half of the data, restricted on the selected features
      if(length(nonzero_index) != 0){
        beta2 <- rep(0, p - 1)
        fit <- lm(y[sample_index2] ~ X[sample_index2, nonzero_index] - 1)
        beta2[nonzero_index] <- as.vector(fit$coeff)
      }
      
      ### calculate the mirror statistics
      M <- abs(beta1 + beta2) - abs(beta1 - beta2)
      selected_index <- analys(M, abs(M), q/2)
      ### the size of the selected neighborhood
      num_select[iter] <- length(selected_index)
      inclusion_rate[iter, selected_index] <- 1/num_select[iter]
    }
    
    ### single data-splitting result
    DS_selected_edge[j, -j] <- ifelse(inclusion_rate[1, ] > 0, 1, 0)
    ### multiple data-splitting result
    inclusion_rate <- apply(inclusion_rate, 2, mean)
    feature_rank <- order(inclusion_rate)
    feature_rank <- setdiff(feature_rank, which(inclusion_rate == 0))
    null_feature <- numeric()
    for(feature_index in 1:length(feature_rank)){
      if(sum(inclusion_rate[feature_rank[1:feature_index]]) > q/2){
        break
      }else{
        null_feature <- c(null_feature, feature_rank[feature_index])
      }
    }
    selected_index <- rep(0, p - 1)
    selected_index[setdiff(feature_rank, null_feature)] = 1
    MDS_selected_edge[j, -j] <- selected_index
  }
  
  ### single data-splitting fdp and power
  fdp_power_result <- fdp_power(DS_selected_edge)
  DS_fdp <- fdp_power_result$fdp
  DS_power <- fdp_power_result$power
  ### multiple data-splitting fdp and power
  fdp_power_result <- fdp_power(MDS_selected_edge)
  MDS_fdp <- fdp_power_result$fdp
  MDS_power <- fdp_power_result$power
  
  return(list(DS_fdp = DS_fdp, DS_power = DS_power, MDS_fdp = MDS_fdp, MDS_power = MDS_power))
}


### GFC-L and GFC-SL (Liu et al 2013)
GFC <- function(data, precision, q){
  fit1 <- SILGGM(data, method = 'GFC_L',  true_graph = precision, alpha = q)
  fit2 <- SILGGM(data, method = 'GFC_SL', true_graph = precision, alpha = q)
  list(fdp = c(fit1$FDR, fit2$FDR), power = c(fit1$power, fit2$power))
}


### BHq and BYq based on pairwise partial correlation test
BH_BY <- function(data, q){
  n <- dim(data)[1]
  p <- dim(data)[2]
  harmonic_sum = sum(1/(1:(p*(p-1)/2)))
  
  ### get pvalues
  pvalues <- NULL
  for(i in 2:p){
    for(j in 1:(i-1)){
      pvalues = suppressWarnings(c(pvalues, pcor.test(data[, i], data[, j], data[, -c(i,j)])$p.value))
    }
  }
  
  ### selected_edge
  sorted_pvalues <- sort(pvalues, decreasing = F, index.return = T)
  BH_cutoff <- max(which(sorted_pvalues$x <= (1:(p*(p-1)/2))*q/(p*(p-1)/2)))
  BY_cutoff <- max(which(sorted_pvalues$x <= (1:(p*(p-1)/2))*q/(p*(p-1)/2*harmonic_sum)))
  BH_selected_edge <- sorted_pvalues$ix[1:BH_cutoff]
  BY_selected_edge <- sorted_pvalues$ix[1:BY_cutoff]
  
  ### calculate fdp and power
  true_edge_set <- c(NULL)
  edge_index <- 0
  for(i in 2:p){
    for(j in 1:(i-1)){
      edge_index <- edge_index + 1
      if(precision[j, i] != 0){
        true_edge_set <- c(true_edge_set, edge_index)
      }
    }
  }
  BH_inc <- intersect(true_edge_set, BH_selected_edge)
  BH_td <- length(BH_inc)
  BH_fdp <- (length(BH_selected_edge) - BH_td)/max(length(BH_selected_edge), 1)
  BH_power <- BH_td/length(true_edge_set)
  
  BY_inc <- intersect(true_edge_set, BY_selected_edge)
  BY_td <- length(BY_inc)
  BY_fdp <- (length(BY_selected_edge) - BY_td)/max(length(BY_selected_edge), 1)
  BY_power <- BY_td/length(true_edge_set)  
  
  return(list(BH_fdp = BH_fdp, BH_power = BH_power, BY_fdp = BY_fdp, BY_power = BY_power))
}


### test out different methods
BH_BY_result <- BH_BY(data, q)
DS_result <- DS(data, q, num_split)
GFC_result <- GFC(data, precision, q)

