from typing import List
from ide.building_blocks.dependency_measure import CMI, MCDE, HiCS, Hoeffdings, dCor, dHSIC, IMIE
from ide.building_blocks.dependency_test import DependencyTest, Kendalltau, XiCor
from ide.building_blocks.dependency_test_adapter import DependencyTestAdapter
from ide.building_blocks.evaluator import DataEfficiencyEvaluator, LogBluePrint
from ide.building_blocks.experiment_modules import DependencyExperiment
from ide.building_blocks.dependency_test import FIT, PeakSim, Pearson, Spearmanr, GMI, DIMID, A_dep_test,chi_square,IndepTest,CondIndTest,LISTest
from ide.building_blocks.selection_criteria import QueryTestNoSelectionCritera
from ide.core.blueprint import Blueprint
from ide.core.data_sampler import DataSampler
from ide.core.evaluator import Evaluator
from ide.core.oracle.augmentation import NoAugmentation
from ide.core.oracle.data_source import DataSource
from ide.core.oracle.oracle import Oracle
from ide.core.query.query_optimizer import NoQueryOptimizer
from ide.modules.data_sampler import KDTreeRegionDataSampler
from ide.modules.evaluator import PlotNewDataPointsEvaluator
from ide.modules.oracle.augmentation import NoiseAugmentation, NoiseConvolution
from ide.modules.oracle.data_source import LineDataSource
from ide.modules.queried_data_pool import FlatQueriedDataPool
from ide.modules.query.query_sampler import LatinHypercubeQuerySampler, RandomChoiceQuerySampler, UniformQuerySampler
from ide.modules.stopping_criteria import LearningStepStoppingCriteria

#Measuring Statistical Dependence with Hilbert-Schmidt Norms (dHSIC) [1]
#Cumulative Mutual Information (CMI)[2]
#Geometric Estimation of Multivariate Dependency (GMI) [3]
#Fast Mutual Information Computation for Dependency-Monitoring on DataStreams (DIMID) [4]
#Iterative Estimation of Mutual Information with Error Bounds (IMIE) [5]
#Peak Similarity (PeakSim) [6]
#Kendall 𝜏 [7]
#Spearman 𝜌 [8]
#Pearson r [9]
#High Contrast Subspaces (HiCS) [10]
#Monte Carlo Dependency Estimation (MCDE) [11]
#New Coefficitent of Correlation (XiCor) [12]
#Fast Conditional Independence Test (FIT) [13]
#Nonparametric tests of independence between random vectors (A.dep.test)[14]
#Hoeffding’s independence test (Hoeffding) [15]
#Correlation of Distance Independence Test (dCOR) [16]
#A Chi-square Test for Conditional Independence (𝜒2) [17]
#Nonparametric Independence Tests Based on Entropy Estimation (IndepTest)[18]
#Nonlinear Conditional Independence Tests (CondIndTests) [19]
#Independence tests for continuous random variables based on the longest increasing subsequence (LISTest) [20]

class BlueprintFactory():
    tests = [
        DependencyTestAdapter(dHSIC()),
        DependencyTestAdapter(CMI()),
        GMI(),
        DIMID(),
        DependencyTestAdapter(IMIE()),
        PeakSim(),
        Kendalltau(),
        Spearmanr(),
        Pearson(),
        DependencyTestAdapter(HiCS()),
        DependencyTestAdapter(MCDE()),
        XiCor(),
        FIT(),
        DependencyTestAdapter(Hoeffdings()),
        DependencyTestAdapter(dCor()),
        IndepTest(),
        CondIndTest(),
        LISTest(),
    ]
    evaluators = [
        DataEfficiencyEvaluator(),
        #PlotNewDataPointsEvaluator()
        #LogBluePrint(),
    ]

    def getBlueprintsForSyntheticData(    
        algorithms: List[DependencyTest] = tests, 
        dataSources: List[DataSource] = [LineDataSource], 
        evaluators: List[Evaluator] = evaluators,
        noiseRatio:float = 0.5
        ):
            blueprints = []
            for dataSource in dataSources:
                for test in algorithms:
                    blueprints.append(Blueprint(
                        repeat=100,
                        stopping_criteria= LearningStepStoppingCriteria(100),
                        queried_data_pool=FlatQueriedDataPool(),
                        initial_query_sampler=UniformQuerySampler(num_queries=10),
                        query_optimizer=NoQueryOptimizer(
                            selection_criteria=QueryTestNoSelectionCritera(),
                            num_queries=10,
                            query_sampler=RandomChoiceQuerySampler(),
                        ),
                        experiment_modules=
                        DependencyExperiment(
                            dependency_test=test,
                        ),
                        oracle = Oracle(
                            data_source=dataSource,
                            augmentation=NoiseAugmentation(noise_ratio=noiseRatio)
                        ),
                        evaluators=evaluators,
                        exp_name = str(test._Configurable__args[0]).replace('(','').replace(')','') + '_noise_' + str(noiseRatio) if isinstance(test,DependencyTestAdapter) else type(test).__name__ + '_noise_' + str(noiseRatio),
                        )
                    )
            return blueprints
 
    def getBlueprintsForRealWorldData(    
        algorithms: List[DependencyTest] = tests, 
        dataSources: List[DataSource] = [LineDataSource], 
        evaluators: List[Evaluator] = evaluators
        ):
            blueprints = []
            for dataSource in dataSources:
                for test in algorithms:
                    blueprints.append(Blueprint(
                        repeat=1,
                        stopping_criteria= LearningStepStoppingCriteria(100),
                    
                        queried_data_pool=FlatQueriedDataPool(),
                        initial_query_sampler=RandomChoiceQuerySampler(num_queries=10),
                        query_optimizer=NoQueryOptimizer(
                            selection_criteria=QueryTestNoSelectionCritera(),
                            num_queries=4,
                            query_sampler=RandomChoiceQuerySampler(),
                        ),
                        experiment_modules=
                        DependencyExperiment(
                            dependency_test=test,
                        ),
                        oracle = Oracle(
                            data_source=dataSource,
                            augmentation=NoAugmentation()
                        ),
                        evaluators=evaluators,
                        exp_name = str(test._Configurable__args[0]).replace('(','').replace(')','') if isinstance(test,DependencyTestAdapter) else type(test).__name__,
                    )
                )
            return blueprints