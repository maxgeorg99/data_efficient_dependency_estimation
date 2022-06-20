#from  ide.experiments.thesis_test_from_measure_synthetic import blueprints
#from ide.experiments.thesis_independent import blueprints
from ide.experiments.thesis_MCDE_synthetic import blueprints
#from ide.experiments.thesis_realWorld import blueprints
#from ide.experiments.thesis_r_synthetic import blueprints
#from ide.experiments.thesis_d_synthetic import blueprints
#from ide.experiments.thesis_classic_synthetic import blueprints
#from ide.experiments.thesis_synthetic import blueprints
from ide.core.experiment_runner import ExperimentRunner

#er = ExperimentRunner([blueprint1,blueprint2,blueprint3])
#er = ExperimentRunner([blueprint])
er = ExperimentRunner(blueprints)
er.run_experiments_parallel()    
