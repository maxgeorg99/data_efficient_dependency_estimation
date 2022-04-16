from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
import numpy as np
import tensorflow as tf

class convergence_evaluator(EvaluationMetric):
    """
    Eveluate convergence of an dependency estimation in an experiment.
    Calculates the Convergence rate. 
    """
    def __init__(self):
        self

    def signal_round_stop(self):
        self.round_number += 1

    def signal_knowledge_discovery_stop(self):
        query = tf.convert_to_tensor(list(range(0,len(self.blueprint.knowledge_discovery_task.sampler.pool.get_all_elements()))))
        xs, ys = self.blueprint.knowledge_discovery_task.surrogate_model.query(query)      

    def get_evaluation(self):
        self