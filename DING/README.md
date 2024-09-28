# Note
- Most of the files used to train and evaluate the DING model are adapted from the DING paper repo: https://github.com/devalab/DING
- Please look at the paper's repo for further instructions on training procedures and model architecture.

# Model file descriptions
- Requirements to run the models can be found in `DING_requirements.txt`.
- the `data` folder contains `roost_unique_elem_dict.pkl` which is simply a list of elements used to train our Roost predictor model.

# Training the model
- `train_generator_script.ipynb` contains the code necessary to train the single property objective DING model.
- `train_generator_script_multiobjective.ipynb` contains the code necessary to train the multi-property objective DING models.
