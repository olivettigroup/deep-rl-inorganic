import pandas as pd
import numpy as np
from pymatgen import Composition
from tqdm import tqdm
from ElM2D import ElM2D

def validate_mats(generated_mats):
    '''
    Checks that each compound is a valid compound, and returns the valid, invalid, 
    and % valid compounds as a decimal.
    Validity: the fraction of a generated materials that are valid.
    
    Args:
    generated_mats: a list of raw generated materials
    Returns:
    valid_mats: a list of valid materials
    invalid_mats: a list of invalid materials
    percent_valid: the percent valid materials as a decimal
    '''
    valid_mats = []
    invalid_mats = []
    for mat in tqdm(generated_mats):
        try:
            comp = Composition(mat).reduced_formula
            valid_mats.append(mat)
        except:
            invalid_mats.append(mat)
    percent_valid = len(valid_mats) / len(generated_mats)
    print(f"Percent valid (as decimal): {round(percent_valid, 4)}")
    return valid_mats, invalid_mats, percent_valid

def standardize_mats(generated_mats):
    '''
    Standardizes generated VALID materials using pymatgen to reduced formula.
    
    Args:
    generated_mats: a list of generated materials
    Returns:
    standardized_mats: a list of standardized materials
    
    '''
    standardized_mats = []
    for mat in tqdm(generated_mats):
        standardized_mats.append(Composition(mat).reduced_formula)
    return standardized_mats

def uniqueness_check(standardized_mats, n_generated=1000):
    '''
    Returns the unique materials and percent uniqueness from standardized generated materials.
    Uniqueness: the fraction of valid generated materials that are unique.
    Standards are typically uniqueness @1k and uniqueness @10k
    
    Args: 
    generated_mats: a list of VALID standardized materials (non-unique). Shape: (sample_size,)
    n_generated: the number of materials that we are comparing the uniqueness check at
    Returns:
    unique_mats: a list of unique generated standardized materials
    percent_unique: the percent unique materials as a decimal
    '''
    assert len(standardized_mats) == n_generated
    unique_mats = list(set(standardized_mats))
    percent_unique = len(unique_mats) / n_generated
    print(f"Percent unique (as decimal) @{n_generated} materials: {round(percent_unique, 4)}")
    return unique_mats, percent_unique
    
def novelty_check(unique_mats, training_set):
    '''
    Returns the novel materials and percent of novel materials that are not in the training set.
    Novelty: the fraction of VALID UNIQUE generated materials that are not in the training set.
    NOTE: unique_mats and training_set should be standardized before passing into this function,
    but we check this anyways here
    
    Args:
    unique_mats: a list of valid unique generated materials. shape = (num_unique,)
    training_set: a list of materials in the training set. shape = (num_test,)
    Returns:
    novel_mats: a list of novel materials.
    percent_novel:  the percent novel materials as a decimal
    '''
    unique_mats = list(set(standardize_mats(unique_mats)))
    training_set = standardize_mats(training_set)
    novel_mats, not_novel_mats = [], []
    for mat in tqdm(unique_mats):
        if mat in training_set:
            not_novel_mats.append(mat)
        else:
            novel_mats.append(mat)
    percent_novel = len(novel_mats) / len(unique_mats)
    print(f"Percent novel (as decimal): {round(percent_novel, 4)}")
    return novel_mats, percent_novel

def similarity_to_nearest_neighbor(unique_mats, metric='mod_petti'):
    '''
    Computes average Element Mover's distance between any two materials in a set of generated materials.
    
    Args:
    unique_mats: a list of valid unique generated materials. shape = (num_unique,)
    NOTE: unique_mats should be standardized before passing into this function,
    but we check this anyways here
    Returns:
    mean_distnace: mean Element Mover's distance between any pair of unique compounds in unique_mats
    std_distance: std Element Mover's distance between any pair of unique compounds in unique_mats
    '''
    # unique_mats = list(set(standardize_mats(unique_mats)))
    unique_mats = standardize_mats(unique_mats)
    print(len(unique_mats))
    mapper = ElM2D(metric)
    mapper.fit(unique_mats)
    distances = []
    for i in tqdm(range(len(unique_mats))):
        for j in range(i+1, len(unique_mats)):
            distances.append(mapper.dm[i][j])
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    print(f"Average Element Mover's distance: {round(mean_distance, 4)}")
    print(f"Std Element Mover's distance: {round(std_distance, 4)}")
    return mean_distance, std_distance
