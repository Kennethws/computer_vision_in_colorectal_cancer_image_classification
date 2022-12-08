#####################################################################
###------------------------------------------------------------------
### Topic : NTU Research Report
### Date  : 2020-06-17
### Author: Chao Wang
###------------------------------------------------------------------
#####################################################################

library(tidyverse)

#--------------------------------------------------------------------
### read data
hmnist64 <- read.csv('data/hmnist_64_64_L.csv', header = T)

#--------------------------------------------------------------------
### wrangle data

x <- hmnist64[,-4097]
## construct different sets of features


# 1. Lower-order histogram features
mu <- rowMeans(x)
variance <- rowSds(as.matrix(x))^2
skew <- apply(x, 1, skewness)
kurt <- apply(x, 1, kurtosis)
m5 <- rowSums((x - mu)^5) / ncol(x)

lower.order <- cbind(mu, variance, skew, kurt, m5) %>% 
  as.data.frame() %>% 
  setNames(c('mean', 'variance', 'skewness', 'kurtosis', 
             'fifth central moment'))


# 2. Higher-order histogram features
higher.order <- sapply(2:11, function(i) {
  rowSums((x - mu)^i) / ncol(x)
})

higher.order <- higher.order %>% 
  as.data.frame() %>% 
  setNames(paste0('central moment ', 2:11))


# 3. Local binary patterns histogram fourier features (LBP-HF)
# LBP
lbp.convert <- function(pic) {
  input <- matrix(pic, 64, 64)
  output <- lbp(input, r = 1)
}

memory.limit(size = 30000)
LBP <- apply(as.matrix(x), 1, lbp.convert)

lbp.uniform <- array(0, c(5000, 62*62))
for (i in 1:5000) {
  lbp.uniform[i,] <- as.vector(LBP[[i]]$lbp.u2)
}

# LBP feature matrix (5000 * 59)
pick <- function(v) {
  out <- array(0, dim = c(1, 59))
  for (i in 1:length(v)) {
    out[v[i] + 1] = out[v[i] + 1] + 1
  }
  out
}

lbp.feature <- apply(lbp.uniform, 1, pick)
lbp.feature <- t(lbp.feature)

# LBP-HF
lbp.hf <- read.csv('data/lbp-hf.csv', header = F)
lbp.hf <- lbp.hf %>% 
  setNames(paste0('LBP', 1:38))

# 4. Local Phase Quantization (LPQ)
lpq.uni <- read.csv('data/LPQ-uniform.csv', header = F)
lpq.uni <- lpq.uni %>% 
  setNames(paste0('LPQ', 1:256))

lpq.gaussian <- read.csv('data/LPQ-gaussian.csv', header = F)
lpq.gaussian <- lpq.gaussian %>% 
  setNames(paste0('LPQ', 1:256))

#--------------------------------------------------------------------
### save data
save(hmnist28, file = 'rdas/hmnist28.rda')
save(hmnist64, file = 'rdas/hmnist64.rda')
save(lower.order, file = 'rdas/lower-order.rda')
save(higher.order, file = 'rdas/higher-order.rda')
save(lbp.uniform, file = 'rdas/lbp-uniform.rda')
save(lbp.feature, file = 'rdas/lbp-feature.rda')
save(LBP, file = 'rdas/lbp.rda')
save(lpq.uni, file = 'rdas/lpq-uniform.rda')
save(lpq.gaussian, file = 'rdas/lpq-gaussian.rda')
save(lbp.hf, file = 'rdas/lbp-hf.rda')
