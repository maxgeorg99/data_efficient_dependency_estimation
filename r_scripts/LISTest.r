install.packages("LISTest",repos = "http://cran.us.r-project.org")
library("LISTest")
data <- read.csv(file = "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/run_data_store/LISTestData.csv")
X1 <- data
X2 <- data
lis.test(X1, X2)