#from ide.experiments.thesis_lin import blueprint
from ide.experiments.de_lin import blueprint
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

er = ExperimentRunner([blueprint])
#er = ExperimentRunner(blueprints)
er.run_experiments()
