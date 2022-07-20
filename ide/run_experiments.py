#from ide.experiments.thesis_realWorld import blueprints
from ide.experiments.thesis_dimensions import blueprints
#from ide.experiments.thesis_synthetic import blueprints
#from ide.experiments.thesis_noise_level import blueprints
#from ide.experiments.thesis_independence import blueprints
from ide.core.experiment_runner import ExperimentRunner

#er = ExperimentRunner([blueprint])
er = ExperimentRunner(blueprints) 
if __name__ == '__main__':  
    er.run_experiments()    
