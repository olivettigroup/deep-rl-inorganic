import sys
import os
os.chdir('/home/jupyter/Elton/RL/DQN')
os.environ["CUDA_VISIBLE_DEVICES"]="1"
sys.path.append('/home/ReLeaSE')
sys.path.append('/home/ReLeaSE/release')
sys.path.append('/home/jupyter/CJK/RL/paper')
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
import numpy as np
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs
from stackRNN import StackAugmentedRNN
from data import GeneratorData
from utils import canonical_smiles
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import re
import pandas as pd
from pymatgen import Composition, Element
from collections import Counter
import json
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from reinforcement import Reinforcement
from data import PredictorData
from utils import get_desc, get_fp
from roost_models.roost_model import predict_formation_energy, predict_bulk_mod, predict_shear_mod, predict_band_gap
from constraints.checkers import check_neutrality, check_electronegativity, check_neutrality_multiple, check_electronegativity_multiple, check_both_multiple
from metrics import similarity_to_nearest_neighbor, standardize_mats
from sklearn.ensemble import RandomForestRegressor
import joblib
from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
import gc
my_predictor = None

# Load RF sinter prediction model
rf_regr = RandomForestRegressor()
rf_regr = joblib.load("rf_models/optimal_sinter_RF.joblib") # final RF model

# Load RF calcine prediction model
rf_regr_calcine = RandomForestRegressor()
rf_regr_calcine = joblib.load("rf_models/optimal_calcine_RF.joblib") # final RF model

# Featurization for RF model
feature_calculators = MultipleFeaturizer([
    cf.element.Stoichiometry(),
    cf.composite.ElementProperty.from_preset("magpie"),
    cf.orbital.ValenceOrbital(props=["avg"]),
    cf.ion.IonProperty(fast=True)
])

# predict sinter temp for a given formula
def predict_sinter(chemical):
    '''
    Predicts the sintering temperature of a material

    Args:
    chemical: Str.
    
    Returns
    sinter_T: float. Predicted sintering temperature
    '''
    chemical = Composition(chemical)
    features = feature_calculators.featurize(chemical)
    features = np.array(features).reshape(1, -1)
    sinter_T = rf_regr.predict(features)[0]
#     except IndexError: # Ad-hoc fix for featurization problem (chemical = Composition(self.state))
#         sinter_T = 1000.0

    return sinter_T

# predict calcine temp for a given formula
def predict_calcine(chemical):
    '''
    Predicts the calcination temperature of a material

    Args:
    chemical: Str.
    
    Returns
    calcine_T: float. Predicted calcination temperature
    '''
    chemical = Composition(chemical)
    features = feature_calculators.featurize(chemical)
    features = np.array(features).reshape(1, -1)
    calcine_T = rf_regr_calcine.predict(features)[0]
#     except IndexError: # Ad-hoc fix for featurization problem (chemical = Composition(self.state))
#         sinter_T = 1000.0

    return calcine_T
    

# function to generate trajectories and receive reward
def estimate_and_update(generator, predictor, prop_to_optimize, prop_to_optimize_2, n_to_generate, gen_data, **kwargs):
    assert prop_to_optimize in ['form_e', 'bulk_mod', 'shear_mod', 'band_gap', 'sinter_temp', 'calcine_temp']
    assert prop_to_optimize_2 in ['form_e', 'bulk_mod', 'shear_mod', 'band_gap', 'sinter_temp', 'calcine_temp']
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated_string = generator.evaluate(gen_data, predict_len=5)
        # remove all starting '<' and ending '>' from generated_strings
        # first char always '<'
        generated_string = generated_string[1:]
        if len(generated_string) > 0:
            if generated_string[-1] == '>':
                # terminated successfully
                generated.append(generated_string[:-1])
            else:
                # got cut off at max length steps
                generated.append(generated_string)

    valid_smiles = []
    prediction = []
    prediction_2 = []
    form_e_prediction = []
    
    pbar = tqdm(generated, total=len(generated), position=0, leave=True)
    for sm in pbar:
        pbar.set_description("Generating predictions...")
        # predict
        try:
            # predict property 1
            c = Composition(sm)
            form_e = predict_formation_energy(sm)
            if prop_to_optimize == "form_e":
                pred = form_e
            elif prop_to_optimize == "bulk_mod":
                pred = predict_bulk_mod(sm)
            elif prop_to_optimize == "shear_mod":
                pred = predict_shear_mod(sm)
            elif prop_to_optimize == "band_gap":
                pred = predict_band_gap(sm)
            elif prop_to_optimize == "sinter_temp":
                pred = predict_sinter(sm)
            elif prop_to_optimize == "calcine_temp":
                pred = predict_calcine(sm)
            # predict property 2
            if prop_to_optimize_2 == "form_e":
                pred_2 = form_e
            elif prop_to_optimize_2 == "bulk_mod":
                pred_2 = predict_bulk_mod(sm)
            elif prop_to_optimize_2 == "shear_mod":
                pred_2 = predict_shear_mod(sm)
            elif prop_to_optimize_2 == "band_gap":
                pred_2 = predict_band_gap(sm)
            elif prop_to_optimize_2 == "sinter_temp":
                pred_2 = predict_sinter(sm)
            elif prop_to_optimize_2 == "calcine_temp":
                pred_2 = predict_calcine(sm)
            prediction.append(pred)
            prediction_2.append(pred_2)
            form_e_prediction.append(form_e)
            valid_smiles.append(sm)
        except:
            # print(f"Invalid compound: {sm}")
            continue
            
    prediction = np.array(prediction) 
    prediction_2 = np.array(prediction_2)
    form_e_prediction = np.array(form_e_prediction)
        
    return valid_smiles, prediction, prediction_2, form_e_prediction

def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

# normalize the rewards since they are on different scales
def normalize(data, prop_to_optimize, minimize=False):
    min_dict = {
        'form_e': -4.334897518157959,
        'bulk_mod': 0,
        'shear_mod': 0,
        'band_gap': 0,
        'sinter_temp': 259.3375,
        'calcine_temp': 367.8357222222222
    }
    max_dict = {
        'form_e': 1.7377617359161377,
        'bulk_mod': 5.673984527587891,
        'shear_mod': 5.663666725158691,
        'band_gap': 7.897870063781738,
        'sinter_temp': 1841.836607142857,
        'calcine_temp': 1597.783333333333
    }
    if minimize:
        return 1 - (data - min_dict[prop_to_optimize])/(max_dict[prop_to_optimize] - min_dict[prop_to_optimize])
    else:
        return (data - min_dict[prop_to_optimize])/(max_dict[prop_to_optimize] - min_dict[prop_to_optimize])

# reward function accepts a single string and generates a prediction for it
def get_reward_max(compound, predictor, prop_to_optimize, prop_to_optimize_2, charge_weight, prop_weight, invalid_reward=-1, get_features=get_fp):
    assert prop_to_optimize in ['form_e', 'bulk_mod', 'shear_mod', 'band_gap', 'sinter_temp', 'calcine_temp']
    assert prop_to_optimize_2 in ['form_e', 'bulk_mod', 'shear_mod', 'band_gap', 'sinter_temp', 'calcine_temp']
    
    elec_charge_reward = 0
    check_charge = False
    check_elec = False
    check_oxide = False
    try:
        check_charge = check_neutrality(compound)
        check_elec = check_electronegativity(compound)
        # check if oxide
        compound_els = list(Composition(compound).get_el_amt_dict().keys())
        if 'O' in compound_els:
            check_oxide = True
    except:
        elec_charge_reward = 0
    if check_elec and check_charge and check_oxide:
        # penalize compounds that aren't charge neutral and electronegativity balanced and oxide
        elec_charge_reward = 1
    
    # if prop_to_optimize == 'form_e':
    #     try:
    #         # predict property of interest
    #         form_e = predict_formation_energy(compound)
    #         form_e = normalize(form_e, prop_to_optimize, minimize=True)
    #         return charge_weight*form_e + (1-charge_weight)*elec_charge_reward
    #     except:
    #         return invalid_reward
    # elif prop_to_optimize == 'bulk_mod':
    #     try:
    #         # predict property of interest
    #         bulk_mod = predict_bulk_mod(compound)
    #         bulk_mod = normalize(bulk_mod, prop_to_optimize, minimize=False)
    #         return charge_charge_weight*bulk_mod + (1-charge_weight)*elec_charge_reward
    #     except:
    #         return invalid_reward
    # elif prop_to_optimize == 'shear_mod':
    #     try:
    #         # predict property of interest
    #         shear_mod = predict_shear_mod(compound)
    #         shear_mod = normalize(shear_mod, prop_to_optimize, minimize=False)
    #         return charge_weight*shear_mod + (1-charge_weight)*elec_charge_reward
    #     except:
    #         return invalid_reward
    # elif prop_to_optimize == 'band_gap':
    #     try:
    #         # predict property of interest
    #         band_gap = predict_band_gap(compound)
    #         band_gap = normalize(band_gap, prop_to_optimize, minimize=False)
    #         return charge_weight*band_gap + (1-charge_weight)*elec_charge_reward
    #     except:
    #         return invalid_reward
    # elif prop_to_optimize == 'sinter_temp':
    #     try:
    #         # predict property of interest
    #         sinter_temp = predict_sinter(compound)
    #         sinter_temp = normalize(sinter_temp, prop_to_optimize, minimize=True)
    #         return charge_weight*sinter_temp + (1-charge_weight)*elec_charge_reward
    #     except: 
    #         return invalid_reward
    # elif prop_to_optimize == 'calcine_temp':
    #     try:
    #         # predict property of interest
    #         calcine_temp = predict_calcine(compound)
    #         calcine_temp = normalize(calcine_temp, prop_to_optimize, minimize=True)
    #         return charge_weight*calcine_temp + (1-charge_weight)*elec_charge_reward
    #     except: 
    #         return invalid_reward
    
    # original
    if prop_to_optimize == "sinter_temp" and prop_to_optimize_2 == "bulk_mod":
        try:
            # predict property of interest
            sinter_temp = predict_sinter(compound)
            sinter_temp = normalize(sinter_temp, prop_to_optimize, minimize=True)
            # predict property 2 of interest
            bulk_mod = predict_bulk_mod(compound)
            bulk_mod = normalize(bulk_mod, prop_to_optimize_2, minimize=False)
            # weigh them accordingly
            prop_reward = prop_weight*sinter_temp + (1-prop_weight)*bulk_mod
            total_reward = charge_weight*prop_reward + (1-charge_weight)*elec_charge_reward
            return total_reward
        except: 
            return invalid_reward

#     if prop_to_optimize == "sinter_temp" and prop_to_optimize_2 == "bulk_mod":
#         try:
#             # predict property of interest
#             sinter_temp = predict_sinter(compound)
#             # predict property 2 of interest
#             bulk_mod = predict_bulk_mod(compound)
#             # weigh them accordingly
#             prop_reward = -1*sinter_temp + prop_weight*bulk_mod
#             total_reward = charge_weight*prop_reward + (1-charge_weight)*elec_charge_reward
#             return total_reward
#         except: 
#             return invalid_reward
    
# uniqueness reward for making sure model doesn't get too focused in a particular subarea
def get_uniqueness_reward(final_compounds):
    # final_compounds_EMD_mean, final_compounds_EMD_std = similarity_to_nearest_neighbor(final_compounds)
    # # normalize
    # final_compounds_EMD_mean /= 30.0
    # return final_compounds_EMD_mean
    standardized_mats = standardize_mats(final_compounds)
    unique_mats = list(set(standardized_mats))
    percent_unique = len(unique_mats) / len(standardized_mats)
    return percent_unique

# main training function
def train_model(prop_to_optimize, prop_to_optimize_2, charge_weight, prop_weight, directory):
#     tracemalloc.start(10)

    with open('/home/jupyter/RL_paper/element_sets/roost_unique_elem_dict.pkl', 'rb') as f:
        roost_elem_dict = pickle.load(f)

    all_elements = roost_elem_dict['mp-bulk-mod.csv']
    all_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # create combination of all elems + all numbers to choose from
    all_elements_numbers = []
    for element in all_elements:
        for number in all_numbers:
            all_elements_numbers.append(element+number)
    tokens = ['<', '>'] + all_elements_numbers

    gen_data_path = '/home/jupyter/CJK/RL/data/mp_all_data_cleaned_oxides_csv.csv'
    gen_data = GeneratorData(training_data_path=gen_data_path, max_len = 5,
                             cols_to_read=[2], keep_header=False, tokens=tokens)


    hidden_size = 1500
    stack_width = 1500
    stack_depth = 10
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta

    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                     output_size=gen_data.n_characters, layer_type=layer_type,
                                     n_layers=3, is_bidirectional=True, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth, 
                                     use_cuda=use_cuda, 
                                     optimizer_instance=optimizer_instance, lr=lr)

    model_path = '/home/ReLeaSE/checkpoints/paper/unbiased/unbiased_10000'
    my_generator.load_model(path=model_path)


    hidden_size = 1500
    stack_width = 1500
    stack_depth = 10
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta

    my_generator_max = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                     output_size=gen_data.n_characters, layer_type=layer_type,
                                     n_layers=3, is_bidirectional=True, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth, 
                                     use_cuda=use_cuda, 
                                     optimizer_instance=optimizer_instance, lr=lr)

    my_generator_max.load_model(model_path)


    # Setting up some parameters for the experiment
    n_to_generate = 200
    n_policy_replay = 10
    n_policy = 15
    n_iterations = 500

    RL_max = Reinforcement(my_generator_max, my_predictor, get_reward_max, get_uniqueness_reward)
    
    # make directory if doesn't exist
    if not os.path.exists(f'/home/ReLeaSE/checkpoints/paper/{directory}'):
        os.makedirs(f'/home/ReLeaSE/checkpoints/paper/{directory}')
    if not os.path.exists(f'/home/ReLeaSE/checkpoints/paper/{directory}/{prop_to_optimize}_{prop_to_optimize_2}'):
        os.makedirs(f'/home/ReLeaSE/checkpoints/paper/{directory}/{prop_to_optimize}_{prop_to_optimize_2}')
    if not os.path.exists(f'/home/ReLeaSE/checkpoints/paper/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}'):
        os.makedirs(f'/home/ReLeaSE/checkpoints/paper/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}')
    if not os.path.exists(f'/home/jupyter/CJK/RL/paper/data/{directory}'):
        os.makedirs(f'/home/jupyter/CJK/RL/paper/data/{directory}')
    if not os.path.exists(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}'):
        os.makedirs(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}')
    if not os.path.exists(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}'):
        os.makedirs(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}')
    if not os.path.exists(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}/compounds'):
        os.makedirs(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}/compounds')

    rewards_max = []
    rl_losses_max = []

    for i in range(1, n_iterations+1):
        for j in trange(n_policy, desc='Policy gradient...', position=0, leave=True):
            cur_reward, cur_loss = RL_max.policy_gradient(gen_data, prop_to_optimize=prop_to_optimize, prop_to_optimize_2=prop_to_optimize_2, 
            charge_weight=charge_weight, prop_weight=prop_weight, get_features=get_fp, n_batch=10)
            rewards_max.append(simple_moving_average(rewards_max, cur_reward)) 
            rl_losses_max.append(simple_moving_average(rl_losses_max, cur_loss))

        # save model every N iterations
        if i == 1 or i % 20 == 0:
            if i == 500 or i == 1000:
                curr_path = f'/home/ReLeaSE/checkpoints/paper/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}/biased_{i}'
                my_generator_max.save_model(curr_path)
            final_compounds, property_trained, property_2_trained, form_e_trained = estimate_and_update(RL_max.generator, my_predictor, prop_to_optimize=prop_to_optimize, 
            prop_to_optimize_2=prop_to_optimize_2, n_to_generate=1000, gen_data=gen_data)
            # save statistics

            # Charge neutrality and electronegativity
            cn_trained_good, cn_trained_bad = check_neutrality_multiple(final_compounds)
            en_trained_good, en_trained_bad = check_electronegativity_multiple(final_compounds)
            # Uniqueness
            final_compounds_unique = list(set(final_compounds))
            # EMD
            final_compounds_EMD_mean, final_compounds_EMD_std = similarity_to_nearest_neighbor(final_compounds)
            # save generated compounds to file
#             with open(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}/compounds/biased_{i}_generated_compounds.pkl', 'wb') as f:
#                 pickle.dump(final_compounds, f)
                
            generated_data_dict = {}
            generated_data_dict['generated_compounds'] = final_compounds
            generated_data_dict[prop_to_optimize] = property_trained
            generated_data_dict[prop_to_optimize_2] = property_2_trained
            generated_data_dict['form_e'] = form_e_trained
            generated_data_dict['cn_trained_good'] = cn_trained_good
            generated_data_dict['cn_trained_bad'] = cn_trained_bad
            generated_data_dict['en_trained_good'] = en_trained_good
            generated_data_dict['en_trained_bad'] = en_trained_bad
            
            with open(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}/compounds/biased_{i}_generated_compound_dict.pkl', 'wb') as f:
                pickle.dump(generated_data_dict, f)
            

            results = {}
            results['iteration'] = i
            results['validity'] = len(final_compounds) / 1000
            results[f'{prop_to_optimize}_trained_mean'] = np.mean(property_trained)
            results[f'{prop_to_optimize}_trained_std'] = np.std(property_trained)
            results[f'{prop_to_optimize_2}_trained_mean'] = np.mean(property_2_trained)
            results[f'{prop_to_optimize_2}_trained_std'] = np.std(property_2_trained)
            results['form_e_trained_mean'] = np.mean(form_e_trained)
            results['form_e_trained_std'] = np.std(form_e_trained)
            results['charge_neutral_trained'] = len(cn_trained_good)/(len(cn_trained_good)+len(cn_trained_bad))
            results['electronegativity_balanced_trained'] = len(en_trained_good)/(len(en_trained_good)+len(en_trained_bad))
            results['unique_trained'] = len(final_compounds_unique)/len(final_compounds)
            results['final_compounds_EMD_mean'] = final_compounds_EMD_mean
            results['final_compounds_EMD_std'] = final_compounds_EMD_std
            results['rewards_max'] = rewards_max
            results['rl_losses_max'] = rl_losses_max

            with open(f'/home/jupyter/CJK/RL/paper/data/{directory}/{prop_to_optimize}_{prop_to_optimize_2}/{str(prop_weight)}/biased_{i}_results.pkl', 'wb') as f:
                pickle.dump(results, f)


#             print('Sample trajectories:')
#             for sm in final_compounds[:5]:
#                 print(sm)
                
            # free results
            del results
            del final_compounds
            del property_trained
            del form_e_trained
            gc.collect()
    # delete models to free memory
    del rewards_max
    del rl_losses_max
    del my_generator_max
    del my_generator
    del gen_data
    del RL_max
    gc.collect()
