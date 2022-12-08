#####################################################################
###------------------------------------------------------------------
### Topic : NTU Research Report
### Date  : 2020-06-17
### Author: Chao Wang
###------------------------------------------------------------------
#####################################################################


library(tidyverse)
library(broom)
library(dslabs)
library(caret)
library(nnet)
library(matrixStats)
library(klaR)
library(xtable)
library(xgboost)
library(e1071)
library(wvtool)
library(ggpubr)
library(MASS)
library(mda)
library(ggrepel)
library(stargazer)
ds_theme_set()
options(digit = 3)

#--------------------------------------------------------------------
### load data
load('rdas/hmnist64.rda')
load('rdas/lower-order.rda')
load('rdas/higher-order.rda')
load('rdas/lbp-feature.rda')
load('rdas/lbp-hf.rda')
load('rdas/lpq-uniform.rda')
load('rdas/lpq-gaussian.rda')

#--------------------------------------------------------------------
### analysis

## display and save sample images
si <- hmnist64 %>% 
  mutate(index = 1:5000) %>% 
  group_by(label) %>% 
  summarise(sample = first(index)) %>% 
  .$sample

tumour <- array(NA, 5000)
tumour[si] <- c('simple-stroma', 'tumour-epithelium', 'complex-stroma',
                'immune-cell', 'debris', 'mucosal-gland', 'adipose-tissue',
                'background')

sapply(si, function(i) {
  png(paste0('figs/',tumour[i],'.png'))
  image(matrix(as.matrix(hmnist64[i, -4097]), 64, 64)[,64:1],
        main = tumour[i],
        xaxt = 'n', yaxt = 'n')
  dev.off()
})

## Preprocessing
y <- factor(hmnist64$label)
x <- hmnist64[,-4097] %>% as.matrix()
x.scale <- apply(x, 2, scale)

  
# column variance
sds <- colSds(x)

qplot(sds, bins = 256, fill = I('red')) + 
  ggtitle('Column SD of each pixel')
ggsave('figs/column-variance.png')

# all features seem to provide some info, can't remove any
nzv <- nearZeroVar(x.scale)


#--------------------------------------------------------------------
## Gray-scale machine learning
# PCA
load('rdas/pca.rda')

# memory.limit(size = 20000)
# pca <- prcomp(x.scale)
# save(pca, file = 'rdas/pca.rda')

set.seed(924)
test.index <- createDataPartition(y, times = 1, p = .1, list = F)
test.x <- pca$x[test.index,]
test.y <- y[test.index]

train.x <- pca$x[-test.index,]
train.y <- y[-test.index]
train.data <- cbind(train.x, train.y)

# plot variance
ten <- first(which(summary(pca)$importance[3,] >= 0.9))
one <- first(which(summary(pca)$importance[3,] >= 0.99))

qplot(1:ncol(train.x), summary(pca)$importance[3,], 
      col = I('blue'), size = I(.7)) +
  xlab('number of columns') +
  ylab('variance retained') +
  ggtitle('variance proportion')
ggsave('figs/variance-proportion.png')

# plot PCs
qplot(PC1, PC2, data = as.data.frame(pca$x), 
      col = y, shape = y) +
  scale_shape_manual(values = 1:nlevels(y)) +
  theme(legend.title = element_blank()) +
  ggtitle('PC2 vs. PC1 for eight categories')
ggsave('figs/plot-pcs.png')

# PCs boxplot
data.frame(type = y, pca$x[, 1:5]) %>% 
  gather(PC, value, -type) %>% 
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot() +
  theme(legend.title = element_blank()) +
  ggtitle('boxplot of 5 PCs')
ggsave('figs/boxplot-5pcs.png')
# only PC1 makes a big difference, which makes sense

# then only plot PC1
data.frame(type = y, pca$x[, 1]) %>% 
  gather(PC, value, -type) %>% 
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot() +
  theme(legend.position = 'none') +
  ggtitle('boxplot of PC1')
ggsave('figs/boxplot-1pc.png')
# looks feasible, can be separated

# extract PCs to retain 90% variance
train.data <- train.data[,c(1:ten, ncol(train.data))]
train.data <- as.data.frame(train.data)
rm(pca, x.scale)

# logistic regression
logist.fit <- multinom(train.y ~ ., data = train.data,
                       maxNWts = 1000000, maxit = 1500)
logist.pred <- predict(logist.fit, test.x)

confusionMatrix(logist.pred, factor(test.y))
logist.table <- confusionMatrix(logist.pred, factor(test.y))$table
logist.class <- confusionMatrix(logist.pred, factor(test.y))$byClass

stargazer(logist.class[,c(1,2,5,7)], title = 'Multinomial Logistic Regression',
          align = T, type = 'text')
stargazer(logist.class[,c(1,2,5,7)], title = 'Multinomial Logistic Regression',
          align = T)


# LDA
memory.limit(size = 20000)
lda.fit <- lda(train.y ~ ., data = train.data)
lda.pred <- predict(lda.fit, test.x)

confusionMatrix(lda.pred, test.y)
lda.table <- confusionMatrix(lda.pred, test.y)$table
lda.class <- confusionMatrix(lda.pred, test.y)$byClass

stargazer(lda.class[,c(1,2,5,7)], title = 'Linear Discriminant Analysis',
          align = T)
stargazer(lda.class[,c(1,2,5,7)], title = 'Linear Discriminant Analysis',
          align = T, type = 'text')


# QDA
qda.fit <- MASS::qda(train.y ~ ., data = train.data)
qda.pred <- predict(qda.fit, as.data.frame(test.x))
mean(qda.pred$class == test.y)

confusionMatrix(qda.pred$class, test.y)
qda.table <- confusionMatrix(qda.pred$class, test.y)$table
qda.class <- confusionMatrix(qda.pred$class, test.y)$byClass

stargazer(qda.class[,c(1,2,5,7)], title = 'Quadratic Discriminant Analysis')
stargazer(qda.class[,c(1,2,5,7)], title = 'Quadratic Discriminant Analysis',
          type = 'text')


# MDA
# install.packages('mda')
library(mda)
mda.fit <- mda(train.y ~ ., data = train.data)
mda.pred <- predict(mda.fit, test.x[,1:ten])

confusionMatrix(mda.pred, test.y)
mda.table <- confusionMatrix(mda.pred, test.y)$table
mda.class <- confusionMatrix(mda.pred, test.y)$byClass

stargazer(mda.class[,c(1,2,5,7)], title = 'Mixture Discriminant Analysis')
stargazer(mda.class[,c(1,2,5,7)], title = 'Mixture Discriminant Analysis',
          type = 'text')


# FDA
library(mda)
fda.fit <- fda(train.y ~ ., data = train.data)
fda.pred <- predict(fda.fit, test.x[,1:ten])

confusionMatrix(fda.pred, test.y)
fda.class <- confusionMatrix(fda.pred, test.y)$byClass

stargazer(fda.class[,c(1,2,5,7)], title = 'Flexible Discriminant Analysis')
stargazer(fda.class[,c(1,2,5,7)], title = 'Flexible Discriminant Analysis',
          type = 'text')


# RDA
# install.packages('klaR')
library(klaR)
rda.fit <- rda(train.y ~ ., data = train.data)
rda.pred <- predict(rda.fit, as.data.frame(test.x))

confusionMatrix(rda.pred$class, test.y)
rda.class <- confusionMatrix(rda.pred$class, test.y)$byClass

stargazer(rda.class[,c(1,2,5,7)], title = 'Regularized Discriminant Analysis')
stargazer(rda.class[,c(1,2,5,7)], title = 'Regularized Discriminant Analysis',
          type = 'text')


## Naive Bayes (NB)
library(klaR)
set.seed(924)
nb.fit <- train(train.y ~ ., data = train.data, method = 'nb',
                trControl = trainControl('cv', number = 10))
nb.pred <- predict(nb.fit, test.x)

confusionMatrix(nb.pred, test.y)
nb.class <- confusionMatrix(nb.pred, test.y)$byClass

stargazer(nb.class[,c(1,2,5,7)], title = 'Naive Bayes')
stargazer(nb.class[,c(1,2,5,7)], title = 'Naive Bayes',
          type = 'text')


## Support Vector Machine (SVM)
# linear SVM
modelLookup('svmLinear')
set.seed(924)
svm.linear.fit <- train(train.y ~ ., data = train.data,
                        method = 'svmLinear',
                        trControl = trainControl('cv', number = 10))
svm.linear.pred <- predict(svm.linear.fit, test.x)

confusionMatrix(svm.linear.pred, test.y)
svm.linear.class <- confusionMatrix(svm.linear.pred, test.y)$byClass

stargazer(svm.linear.class[,c(1,2,5,7)], title = 'Linear SVM')
stargazer(svm.linear.class[,c(1,2,5,7)], title = 'Linear SVM',
          type = 'text')

# nonlinear SVM
# radial basis kernel
set.seed(924)
svm.rb.fit <- train(train.y ~ ., data = train.data,
                    method = 'svmRadial', 
                    trControl = trainControl('cv', 10))
svm.rb.pred <- predict(svm.rb.fit, test.x)

confusionMatrix(svm.rb.pred, test.y)
svm.rb.class <- confusionMatrix(svm.rb.pred, test.y)$byClass

stargazer(svm.rb.class[,c(1,2,5,7)], title = 'Radial Basis SVM')
stargazer(svm.rb.class[,c(1,2,5,7)], title = 'Radial Basis SVM',
          type = 'text')

## Deep Neural Network (DNN) - kinda failure
# deepnet package
# install.packages('deepnet')
# install.packages("neuralnet")
# install.packages("h2o")
library(deepnet)
library(neuralnet)
library(h2o)

# neuralnet
train.data$train.y <- as.factor(train.data$train.y)
nn.fit <- neuralnet(train.y ~ ., data = train.data,
                    hidden = c(5), act.fct = 'logistic',
                    linear.output = F)
nn.pred <- compute(nn.fit, test.x[,1:ten])

nn.result <- nn.pred$net.result
nn.result %>% head
nn.pred <- apply(nn.result, 1, which.max) %>% factor()
confusionMatrix(nn.pred, test.y)

localh2o <- h2o.init(ip = 'localhost', port = 54321,
                     startH2O = T, nthreads = -1)

dnn.train <- h2o.importFile('data/dnn-train.csv')
dnn.test <- h2o.importFile('data/dnn-test.csv')

y <- 'C1'
x <- setdiff(names(dnn.test),y)
dnn.train[,y] <- as.factor(dnn.train[,y])
dnn.test[,y] <- as.factor(dnn.test[,y])

model <- h2o.deeplearning(x = x,
                          y = y,
                          training_frame = dnn.train,
                          validation_frame = dnn.test,
                          distribution = "multinomial",
                          activation = "RectifierWithDropout",
                          hidden = c(10,10),
                          input_dropout_ratio = 0.2,
                          l1 = 1e-5,
                          epochs = 20)


#--------------------------------------------------------------------
## machine learning
low.scale <- lower.order %>% 
  mutate_all(scale)

high.scale <- higher.order %>% 
  mutate_all(scale)

lbp.scale <- lbp.feature %>% 
  as.data.frame() %>% 
  mutate_all(scale)

lbp.hf.scale <- lbp.hf %>% 
  mutate_all(scale)

lpq.uni.scale <- lpq.uni %>% 
  mutate_all(scale)

lpq.gaussian.scale <- lpq.gaussian %>% 
  mutate_all(scale)

# LPQ PCA
lpq.pca <- prcomp(lpq.gaussian.scale)
ten <- first(which(summary(lpq.pca)$importance[3,] > 0.9))
five <- first(which(summary(lpq.pca)$importance[3,] > 0.95))
twenty <- first(which(summary(lpq.pca)$importance[3,] > 0.8))
thirty <- first(which(summary(lpq.pca)$importance[3,] > 0.7))
fourty <- first(which(summary(lpq.pca)$importance[3,] > 0.6))

set.seed(924)
shuffle <- sample(nrow(hmnist64))
x <- cbind(lpq.pca$x[, 1:thirty], low.scale, high.scale, lbp.hf.scale)[shuffle,]
y <- hmnist64$label[shuffle]
# if don't use factor(), the accuracy will be much higher after
# createDataPartition() but the result is less interpretable

set.seed(924)
test.index <- createDataPartition(y, times = 1, p = .1, list = F)
test.x <- x[test.index,]
test.y <- y[test.index] %>% factor()

train.x <- x[-test.index,]
train.y <- y[-test.index] %>% factor()
train.data <- cbind(train.x, train.y)

# logistic
logist.fit <- multinom(train.y ~ ., data = train.data, 
                       maxit = 2000, MaxNWts = 10000000)
logist.pred <- predict(logist.fit, test.x)
mean(logist.pred == test.y)

confusionMatrix(logist.pred, test.y)

logist.table <- confusionMatrix(logist.pred, test.y)$table
logist.class <- confusionMatrix(logist.pred, test.y)$byClass

# Latex output
xtable(logist.table, caption = 'Multinomial Logistic Regression')
stargazer(logist.class[,c(1,2,5,7)], title = 'Multinomial Logistic Regression')


# SVM
# linear
load('rdas/svm-linear.rda')
svm.linear.fit <- train(train.x, train.y, method = 'svmLinear',
                 trControl = trainControl('cv', number = 10),
                 tuneGrid = expand.grid(C = 1))
svm.linear.fit$bestTune

ggplot(svm.linear.fit, highlight = T)
# save(svm.linear.fit, file = 'rdas/svm-linear.rda')

svm.linear.pred <- predict(svm.linear.fit, test.x)
mean(svm.linear.pred == test.y)

confusionMatrix(svm.linear.pred, test.y)

# learning curve
svm.linear.data <- learning_curve_dat(train.data, outcome = 'train.y',
                                      method = 'svmLinear',
                                      trControl = trainControl('cv', 10),
                                      tuneGrid = expand.grid(C = 1))

p1 <- svm.linear.data %>% 
  ggplot(aes(Training_Size, 1 - Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  ylab('Error') +
  ggtitle('Learning Curve of Linear SVM')
p1
ggsave(filename = 'figs/linear-svm-learning-curve.png')

# radial basis
load('rdas/svm-rb.rda')
modelLookup('svmRadial')
svm.rb.fit <- train(train.x, train.y, method = 'svmRadial',
                    trControl = trainControl('cv', number = 10),
                    tuneLength = 1)
svm.rb.fit$bestTune

ggplot(svm.rb.fit, highlight = T)
# save(svm.rb.fit, file = 'rdas/svm-rb.rda')

svm.rb.pred <- predict(svm.rb.fit, test.x)
mean(svm.rb.pred == test.y)

confusionMatrix(svm.rb.pred, test.y)

# learning curve
svm.rb.data <- learning_curve_dat(train.data,
                                  outcome = 'train.y',
                                  trControl = trainControl('cv', 10),
                                  method = 'svmRadial',
                                  tuneGrid = expand.grid(sigma = 0.004502533,
                                                         C = 2))

p2 <- svm.rb.data %>% 
  ggplot(aes(Training_Size, 1 - Accuracy, col = Data)) +
  geom_smooth(method = loess, span = .8) +
  ylab('Error') +
  ggtitle('Learning Curve of rb SVM')
p2
ggsave(filename = 'figs/rb-svm-learning-curve.png')
# high var, decrease C

# polynomial
load('rdas/svm-poly.rda')
modelLookup('svmPoly')
svm.poly.fit <- train(train.x, train.y, method = 'svmPoly',
                      trControl = trainControl('cv', 10),
                      tuneGrid = expand.grid(degree = 2,
                                             scale = .01,
                                             C = .5))
svm.poly.fit$bestTune

ggplot(svm.poly.fit, highlight = T)
# save(svm.poly.fit, file = 'rdas/svm-poly.rda')

svm.poly.pred <- predict(svm.poly.fit, test.x)
mean(svm.poly.pred == test.y)

confusionMatrix(svm.poly.pred, test.y)


# random forest
# need to tune parameters
load('rdas/rf-fit.rda')
rf.fit <- train(train.x, train.y, method = 'Rborist',
                trControl = trainControl('cv', 10),
                tuneGrid = expand.grid(predFixed = 28,
                                       minNode = 2))
rf.fit$bestTune

ggplot(rf.fit, highlight = T)
# save(rf.fit, file = 'rdas/rf-fit.rda')

rf.pred <- predict(rf.fit, test.x)
mean(rf.pred == test.y)

confusionMatrix(rf.pred, test.y)

# learning curve
rf.data <- learning_curve_dat(train.data, outcome = 'train.y',
                              method = 'Rborist',
                              trControl = trainControl('cv', 10),
                              tuneGrid = expand.grid(predFixed = 28,
                                                     minNode = 2))

p3 <- rf.data %>% 
  ggplot(aes(Training_Size, 1 - Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  ylab('Error') +
  ggtitle('Learning Curve of Random Forest')
p3
ggsave(filename = 'figs/randon-forest-learning-curve.png')

# aligned plots
ggarrange(p1, p2, p3, 
          labels = c('A', 'B', 'C'),
          common.legend = T, legend = 'bottom')

# XGBoost
# need to tune parameters
load('rdas/xg-fit.rda')
xg.fit <- train(train.x, train.y, method = 'xgbTree',
                trControl = trainControl('cv', 10),
                tuneLength = 2)

xg.fit <- train(train.x, train.y, method = 'xgbTree',
                trControl = trainControl('cv', 10),
                tuneGrid = expand.grid(nrounds = 100,
                                       max_depth = 2,
                                       eta = .4,
                                       gamma = 0,
                                       colsample_bytree = .8,
                                       min_child_weight = 1,
                                       subsample = 1))


xg.fit$bestTune

xg.pred <- predict(xg.fit, test.x)
mean(xg.pred == test.y)
# save(xg.fit, file = 'rdas/xg-fit.rda')

confusionMatrix(xg.pred, test.y)

# learning curve
xg.fit$bestTune


xg.data <- matrix(0, ncol = 3, nrow = 10)
colnames(xg.data) <- c('size', 'training', 'cv')
for (i in 1:10) {
  # train model
  xg <- train(train.y ~ ., data = train.data[sample(nrow(train.data), 400*i),],
              method = 'xgbTree',
              trControl = trainControl('cv', 10),
              nrounds = 150, max_depth = 4,
              eta = .3, gamma = 0,
              colsample_bytree = .6, 
              min_child_weight = 1,
              subsample = 1)
  xg.data[i, 1] <- 500 * i
  xg.data[i, 2] <- xg$results$Accuracy
  
  # prediction
  pred <- predict(xg, test.x)
  xg.data[i, 3] <- mean(pred == test.y)
}

xg.data <- as.data.frame(xg.data)

# Ensemble
most <- function(v) {
  tab <- table(v)
  names(tab)[which.max(tab)]
}

ensemble <- cbind(logist.pred, rf.pred, xg.pred)

ensemble.pred <- apply(ensemble, 1, most) %>% factor()
mean(ensemble.pred == test.y)
# now is already more accurate than radial basis SVM alone!

confusionMatrix(ensemble.pred, test.y)

ensemble.table <- confusionMatrix(ensemble.pred, test.y)$table
ensemble.class <- confusionMatrix(ensemble.pred, test.y)$byClass

xtable(ensemble.table)

stargazer(ensemble.class[,c(1,2,5,7)], title = 'Ensemble of Best 3',
          type = 'text')
stargazer(ensemble.class[,c(1,2,5,7)], title = 'Ensemble of Best 3')

# histogram all 4
h <- data.frame(method = c('MLR', 'Lin SVM', 'rb SVM', 
                             'Poly SVM', 'RF', 'XGBoost',
                             'Ensemble'),
                  error = 100 - c(86.4, 85.6, 84.2, 83.6, 88,
                               90, 90.6))

h %>% 
  mutate(method = reorder(method, error), error = round(error, 1)) %>% 
  ggplot(aes(method, error, fill = method)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(label = error), vjust = -.3, color = 'black',
            size = 3.5) +
  scale_fill_brewer(palette = 'Reds') +
  ylab('Error Rate (%)') +
  xlab('Algorithm') +
  ggtitle('Algorithm Error Rate') +
  theme(legend.position = 'bottom') +
  ylim(0, 25)
ggsave('figs/best4.png')

# best 4 vs. best 3
h2 <- data.frame(method = c('MLR', 'Lin SVM', 'rb SVM', 
                            'Poly SVM', 'RF', 'XGBoost',
                            'Ensemble',
                            'MLR', 'Lin SVM', 'rb SVM', 
                            'Poly SVM', 'RF', 'XGBoost',
                            'Ensemble'),
                 error = 100 - c(86.8, 87.4, 80.4, 81.8, 88.4,
                                 89.2, 90.8,
                                 86.4, 85.6, 84.2, 83.6, 88,
                                 90, 90.6),
                 type = c(rep('Best3', 7), rep('Best4', 7)))

h2 %>% 
  mutate(method = reorder(method, error), error = round(error, 1)) %>% 
  ggplot(aes(x = method, y = error)) +
  geom_col(aes(color = type, fill = type),
           position = position_dodge(0.8), width = .7) +
  scale_color_manual(values = c('#0073C2FF', '#EFC000FF')) +
  scale_fill_manual(values = c('#0073C2FF', '#EFC000FF')) +
  ylim(0, 24) +
  geom_text(aes(label = error, group = type), position = position_dodge(.8),
            vjust = -.3, size = 3) +
  xlab('Algorithm') +
  ylab('Error Rate (%)') +
  ggtitle('Best 4 vs. Best 3') +
  theme(legend.position = 'bottom')
ggsave('figs/comparison-best4.png')

# wide format
h.pure <- data.frame(feature = c('Lower', 'Higher', 'LBP-HF',
                                 'LPQ'),
                     MLR = 100 - c(69.4, 65.2, 72.2, 70.6),
                     LinSVM = 100 - c(70.6, 63.8, 71.8, 71.8),
                     rbSVM = 100 - c(78, 64.4, 68.6, 69.8),
                     PolySVM = 100 - c(58.4, 46.4, 68.6, 72.4),
                     RF = 100 - c(77.6, 68, 73.8, 75),
                     XGBoost = 100 - c(78.4, 66.6, 74.2, 79),
                     Ensemble = 100 - c(79.2, 69.6, 75.2, 79.8))

# wide to long
h.pure <- h.pure %>% 
  gather(method, error, -feature)

h.pure %>% 
  ggplot(aes(x = feature, y = error, fill = method)) +
  geom_bar(stat = 'identity', position = position_dodge(.8),
           width = .7) +
  theme(legend.position = 'bottom', legend.box = 'horizontal') +
  scale_fill_brewer(palette = 'Blues', 
                    name = 'Algorithm') +
  geom_text(aes(label = error),
            vjust = -.5, size = 2.5,
            position = position_dodge(.8)) +
  xlab('Pure Feature') +
  ylab('Error Rate (%)') +
  ylim(0, 58) +
  ggtitle('Pure Feature Set Error Rate')
ggsave('figs/pure-feature.png', width = 8.33, height = 3.22)

# best 1 to 4 overview
h4 <- data.frame(method = c('My Model', 'XGBoost', 'Random Forest', 'Multi-Logistic', 
                            'LinSVM', 'PolySVM', 'GaussSVM'),
                 error = c(20.2, 21, 25, 29.4, 28.2, 27.6, 30.2,
                           10, 10.2, 11.6, 16.2, 18.8, 21, 20.2,
                           9.2, 10.8, 11.6, 13.2, 12.6, 18.2, 19.6,
                           9.4, 10, 12, 13.6, 14.4, 16.4, 15.8),
                 Combination = c(rep("Best Single Set: 80% LPQ", 7), rep("Best 2: 70% LPQ + Lower", 7), rep('Best 3: Best 2 + Higher', 7), rep('All 4: Best 3 + LBP', 7)))

h4 %>% 
  mutate(method = factor(method, levels = unique(h4$method)), error = round(error, 1), Combination = factor(Combination, levels = unique(Combination))) %>%
  ggplot(aes(x = method, y = error)) +
  geom_col(aes(color = Combination, fill = Combination),
           position = position_dodge(0.8), width = .7) +
  scale_color_manual(values = c('#bdd7e7', '#6baed6', '#3182bd', '#08519c')) +
  scale_fill_manual(values = c('#bdd7e7', '#6baed6', '#3182bd', '#08519c')) +
  ylim(0, 34) +
  geom_text(aes(label = error, group = Combination), position = position_dodge(.8),
            vjust = -.3, size = 3) +
  xlab('Algorithm') +
  ylab('Error Rate (%)') +
  ggtitle('Prediction error rates (1 - accuracy) of best combinations of feature sets across different algorithms') +
  theme(legend.position = 'bottom')
ggsave('figs/mit-slides.png', width = 10, height = 4)
