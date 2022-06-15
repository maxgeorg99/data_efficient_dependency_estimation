install.packages("IndepTest",repos = "http://cran.us.r-project.org",lib = "C:/Users/maxig/OneDrive/Dokumente/R/win-library/4.1")
library("IndepTest",lib.loc = "C:/Users/maxig/OneDrive/Dokumente/R/win-library/4.1")
data <- read.csv(file = "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/run_data_store/indepTestData.csv")
MINTauto(data,data,kmax=50,B1=100,B2=100)