list.of.packages <- c("LIStest")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages,repos = "http://cran.us.r-project.org",lib = "C:/Users/maxig/OneDrive/Dokumente/R/win-library/4.1")
library("LIStest",lib.loc = "C:/Users/maxig/OneDrive/Dokumente/R/win-library/4.1")
data <- read.csv(file = "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/run_data_store/LISTestData.csv")
X1 <- unlist(data[,1])
X2 <- unlist(data[,2])
lis.test(X1, X2)