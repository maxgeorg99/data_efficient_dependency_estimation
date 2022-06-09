#from ide.experiments.thesis_lin import blueprint
#from ide.experiments.de_lin import blueprint
#from ide.experiments.thesis_java import blueprint1
#from ide.experiments.thesis_java import blueprint2
#from ide.experiments.thesis_java import blueprint3
#from  ide.experiments.thesis_test_from_measure_synthetic import blueprints
#from ide.experiments.de_sqr import blueprint
#from ide.experiments.ide_lin import blueprint
#from ide.experiments.ide_grid_blueprint import blueprint
#from ide.experiments.optimal_query_lin import blueprint
#from ide.experiments.optimal_query_sqr import blueprint
#from ide.experiments.de_hourglas import blueprint
#from ide.experiments.de_crossdata import blueprint
from ide.experiments.thesis_synthetic import blueprints
#from ide.experiments.thesis_realWorld import blueprints
from ide.core.experiment_runner import ExperimentRunner

#er = ExperimentRunner([blueprint1,blueprint2,blueprint3])
#er = ExperimentRunner([blueprint])
er = ExperimentRunner(blueprints)
er.run_experiments()    
