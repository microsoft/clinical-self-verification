This is an evolving repo optimized for machine-learning projects aimed at designing a new algorithm. They require sweeping over different hyperparameters, comparing to baselines, and iteratively refining an algorithm. Based of [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science).

# Organization
- `project_name`: should be renamed, contains main code for modeling (e.g. model architecture)
- `experiments`: code for runnning experiments (e.g. loading data, training models, evaluating models)
- `scripts`: scripts for running experiments (e.g. python scripts that launch jobs in `experiments` folder with different hyperparams)
- `notebooks`: jupyter notebooks for analyzing results and making figures
- `tests`: unit tests

# Setup
- first, rename `project_name` to your project name and modify `setup.py` accordingly
- clone and run `pip install -e .`, resulting in a package named `project_name` that can be imported
    - see `setup.py` for dependencies, not all are required
- example run: run `python scripts/01_train_basic_models.py` (which calls `experiments/01_train_model.py` then view the results in `notebooks/01_model_results.ipynb`
- keep tests upated and run using `pytest`

# Features
- scripts sweep over hyperparameters using easy-to-specify python code
- experiments automatically cache runs that have already completed
    - caching uses the (**non-default**) arguments in the argparse namespace
- notebooks can easily evaluate results aggregated over multiple experiments using pandas

# Guidelines
- See some useful packages [here](https://csinva.io/blog/misc/ml_coding_tips)
- Avoid notebooks whenever possible (ideally, only for analyzing results, making figures)
- Paths should be specified relative to a file's location (e.g. `os.path.join(os.path.dirname(__file__), 'data')`)
- Naming variables: use the main thing first followed by the modifiers (e.g. `X_train`, `acc_test`)
    - binary arguments should start with the word "use" (e.g. `--use_caching`) and take values 0 or 1
- Use logging instead of print
- Use argparse and sweep over hyperparams using python scripts (or custom things, like [amulet](https://amulet-docs.azurewebsites.net/main/index.html))
    - Note, arguments get passed as strings so shouldn't pass args that aren't primitives or a list of primitives (more complex structures should be handled in the experiments code)
- Each run should save a single pickle file of its results
- All experiments that depend on each other should run end-to-end with one script (caching things along the way)
- Keep updated requirements in setup.py
- Follow sklearn apis whenever possible
- Use Huggingface whenever possible, then pytorch