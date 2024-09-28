# Model file descriptions
- Requirements to run the models can be found in `pgn_requirements.txt`.
- The `data` folder contains `data/mp_all_data_cleaned_oxides_csv.csv` file, which is a CSV file with all of the downloaded data from Materials Project for the oxides used to pre-train the PGN model. This includes important information such as formula, mp-id, and materials properties such as band gap, shear modulus, bulk modulus, and formation energy per atom.
- `data.py` is a data loader class that is configured to import inorganic material formula data from a specified file path. Pass in a `training_data_path` which points to your training dataset, in this case we use `mp_all_data_cleaned_oxides_csv.csv`.
- `utils.py` contains utility functions for model training and evaluation. The most important function is `get_fp` which featurizes inorganic formulas using Magpie features.
- `reinforcement.py` contains the code for executing the RL training loop. The most important function to pass in is the reward function `get_reward`, which is defined and passed in from the training notebooks listed below.
`stackRNN.py` contains the model architecture for the stack RNN (`StackAugmentedRNN`) used for material generation.


# Training the models
- `train_RNN_MP_oxides_unbiased_save_iteratively.ipynb` contains the code necessary to train the unbiased RNN generator. This resulting pre-trained model will be used as the base model to be biased using reinforcement learning in subsequent steps.
    - Set `gen_data_path` to your data to use to pretrain the model, which in this case we use `mp_all_data_cleaned_oxides_csv.csv`.
    - You will also need an element dictionary to create an element action space for the model. In this case we use `roost_unique_elem_dict.pkl` which is simply a list of elements used to train our Roost predictor model and can be found in the notebook.
    - By default, the model will be saved every 1000 epochs. This can be configured by the user.
-  `train_RL_models_hyperparameter.py` contains the code to train the biased RNN generator through RL.
    - You will have to specify a `rf_regr` and `rf_regr_calcine` models, which are random forest models trained to predict sintering and calcination temperatures, respectively.
    - The `train_model(prop_to_optimize, weight, directory)` function allows the user to specify a property to optimize, a weight to weigh the charge/electronegativity balance rewards with the property reward (typically 0.5 is a good setting), and a directory path to save files in.
    - By default, the model will be saved every 1000 epochs. This can be configured by the user.
- `train_RL_models_hyperparameter_multiobjective.py` contains the code to train the biased RNN generator through RL with multiple property objectives.
    - You will have to specify a `rf_regr` and `rf_regr_calcine` models, which are random forest models trained to predict sintering and calcination temperatures, respectively.
    - The `train_model(prop_to_optimize, weight, directory)` function allows the user to specify a property to optimize, a weight to weigh the charge/electronegativity balance rewards with the property reward (typically 0.5 is a good setting), and a directory path to save files in.
    - By default, the model will be saved every 1000 epochs. This can be configured by the user.