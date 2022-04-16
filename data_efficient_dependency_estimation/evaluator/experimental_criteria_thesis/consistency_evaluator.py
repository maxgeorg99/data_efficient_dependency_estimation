from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
import numpy as np
import tensorflow as tf

class consistency_evaluator(EvaluationMetric):
    """
    Eveluate concistency in an experiment.
    Compare uncertainties with different starting samples.
    """
    def __init__(self):
        self.round_number = -1

    def signal_round_stop(self):
        self.round_number += 1

    def signal_knowledge_discovery_stop(self):
        """
        Get Uncertainity for random sample
        """
        query = tf.convert_to_tensor(list(range(0,len(self.blueprint.knowledge_discovery_task.sampler.sample()))))
        result = self.blueprint.knowledge_discovery_task.uncertainty(query)
        self.results.append(result)

    def eval(self):
        """
        Compare uncertainties
        """
        print(self.results[self.round_number])

    def get_evaluation(self):
        return self.results