from sklearn.metrics import f1_score, roc_auc_score
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from data_efficient_dependency_estimation.knowledge_discovery.dependency_tests_thesis import mcde
import numpy as np
import tensorflow as tf
from statsmodels.stats.power import TTestIndPower

class data_efficiency_evaluator(EvaluationMetric):
    """
    Measures data efficiency metrics of an dependency estimation in an experiment.
    The metric includes 
    * Statistical Power
    * Distribution of dependency estimation scores
    * F1
    * AUC
    in propotion to the amount of samples.
    """
    effect: None
    power: None
    alpha: None
    y_true: None

    def __init__(self):
        self.round_number = -1
        self.evaluation_metrics = {'Power', 'Dependency score','F1','AUC'}
        self.results = dict().fromkeys(self.evaluation_metrics)
        "calculate GroundTruth for DataSet with MCDE"
        self.y_true = mcde.MonteCarloDependencyEstimation.learn(100)

    def signal_round_stop(self):
        self.round_number += 1

    def signal_knowledge_discovery_stop(self):
        """
        calculate metrics
        """
        result = self.blueprint.knowledge_discovery_task.learn(100)
        self.evaluation_metrics['Power'] = TTestIndPower().solve_power(self.effect, power=self.power, nobs1=None, ratio=1.0, alpha=self.alpha)
        self.evaluation_metrics['Dependency score'] = result.p
        self.evaluation_metrics['F1'] = f1_score(self.y_true, result)
        self.evaluation_metrics['AUC'] = roc_auc_score(self.y_true, result)
        self.results.append(result)

    def eval(self):
        print(self.results[self.round_number])

    def get_evaluation(self):
        return self.results