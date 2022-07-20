list.of.packages <- c("FNN", "mvtnorm","IndepTest")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages,repos = "http://cran.us.r-project.org",lib = "C:/Users/maxig/OneDrive/Dokumente/R/win-library/4.1")
library("FNN",lib.loc = "C:/Users/maxig/OneDrive/Dokumente/R/win-library/4.1")
library("mvtnorm",lib.loc = "C:/Users/maxig/OneDrive/Dokumente/R/win-library/4.1")
library("IndepTest",lib.loc = "C:/Users/maxig/OneDrive/Dokumente/R/win-library/4.1")
data <- read.csv(file = "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/run_data_store/indepTestData.csv")
n <- ncol(data)
MINTauto(data[,1:(n-1)],data[,n],kmax=50,B1=100,B2=100)