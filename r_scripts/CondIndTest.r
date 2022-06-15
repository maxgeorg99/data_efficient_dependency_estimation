install.packages("CondIndTests",repos = "http://cran.us.r-project.org")
library("CondIndTests")
data <- read.csv(file = "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/run_data_store/condIndTestData.csv")
X <- data
Y <- data
Z  <- as.list(rep(1, length(data)))
test <- CondIndTest(X, Y, Z, method = "KCI")
print(test$pvalue)