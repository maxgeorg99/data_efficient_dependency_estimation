For running all the experiments in this project a working version of Python 3.9.6, java and R has to be installed on the machine.

To install this project on your machine clone the repository by executing

```sh
git clone https://github.com/maxgeorg99/data_efficient_dependency_estimation
```

and execute the poetry (https://python-poetry.org/) command 

```sh
poetry install
```

To build a  virtual environment execute the poetry shell command.

```sh
poetry shell
```

The experiments are defined as blueprints in the experiments folder.

Real world data can be downloaded from here: https://efss.qloud.my/index.php/s/oHybxZcjKRJo74N 
you can than unzip the real_world_data_sets folder into the project folder.

To add additional experiments add a new python file in the experiment folder where you define the blueprint or multiple blueprints and specify the experiment parameters.

Start the execution of the experiments by executing the run_experiments.py file.

You can choose which experiment to execute by importing the blueptrints from the experiments.

After the execution run the desired post experiment script from the post_experiment_computaion folder.

For the results of the calculate_gain script you have to run the algorithms class by class and define the baseline in the calculate_gain.py script.

The output graphics are placed in the experiment_results or fig folder.
