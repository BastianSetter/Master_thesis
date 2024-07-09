import logging as lg
import numpy as np
import pandas as pd
from dataclasses import dataclass
from flatten_list.flatten_list import flatten


#--------------Generel-----------------
def remove_suffix(text, suffix):
    pos = text.rfind(suffix)
    if pos != -1 and pos == len(text) - len(suffix):
        return text[:pos]
    return text

def recursive_sum(lst):
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += recursive_sum(item)
        elif isinstance(item, (int, float)):
            total += item
    return total

def get_mask_for_du_ptype_cc_on_df(y_df:pd.DataFrame, dataset_union=None, pdgid=None, current=None, interaction=None):
    mask = np.ones(shape=len(y_df),dtype=bool)
    if dataset_union is not None:
        if not pd.api.types.is_list_like(dataset_union):
            dataset_union = [dataset_union]
        mask &= y_df['dataset'].isin(dataset_union)

    if pdgid is not None:
        if not pd.api.types.is_list_like(pdgid):
            pdgid = [pdgid]
        mask &= y_df["particle_type"].isin(pdgid)

    if current is not None:
        if not pd.api.types.is_list_like(current):
            current = [current]
        mask &= y_df["current"].isin(current)

    if interaction is not None:#TODO revisit isin (change may be due to other error)
        if not pd.api.types.is_list_like(interaction):
            interaction = [interaction]

        mask_i = np.zeros_like(mask, dtype=bool)
        for itype in interaction:
            mask_i |= (y_df['particle_type'].astype(int) == int(itype.pdgid)) & (y_df['current'] == itype.current)

        mask &= mask_i
    return mask

#--------------Dataset-----------------
def dset_tree_to_str(lst):
    total = ''
    for item in lst:
        if isinstance(item, list):
            total += f'<{dset_tree_to_str(item)}>'
        else:
            total += item
        total += '_'
    return total[:-1]

    
dset_name_to_id = {
    'lukas_flat': 0,
    'standard_v7.1': 7.1,
    'standard_v7.2': 7.2
}

dset_id_to_name = {value: key for key, value in dset_name_to_id.items()}

def get_all_datasets_from_all_unions(unions):
    flat_datasets = list(flatten(unions))
    return flat_datasets

#------------Interaction---------------
@dataclass
class Interaction:
    pdgid: int
    current: str

def itype_tree_to_str(lst):
    total = ''
    for item in lst:
        if isinstance(item, list):
            total += f'<{itype_tree_to_str(item)}>'
        elif isinstance(item, Interaction):
            total += interaction_to_str(item)
        total += '_'
    return total[:-1]

def interaction_to_str(itype:Interaction):
    return f'{itype.pdgid}-{itype.current}'

def _extend_unions(unions):
    return unions


def convert_to_interaction(data):
    data = data.copy()
    try:
        data[0] = int(data[0])
    except TypeError:
        pass
    
    if len(data) == 2 and isinstance(data[0], int) and isinstance(data[1], str):
        return Interaction(pdgid=data[0], current=data[1])
    elif pd.api.types.is_list_like(data):
        return [convert_to_interaction(item) for item in data]
    else:
        raise ValueError("Unsupported data format")

def interactions_to_tuples(interactions):
    if isinstance(interactions, Interaction):
        return (interactions.pdgid, interactions.current)
    elif isinstance(interactions, list):
        return [interactions_to_tuples(interaction) for interaction in interactions]

def get_all_interactions_from_all_unions(unions):
    flat_interactions = list(flatten(unions))
    return flat_interactions

def _unique_interactions_in_df(df:pd.DataFrame):
        unique_rows_df = df.drop_duplicates(subset=['particle_type','current'])
        unique_rows_list = unique_rows_df[['particle_type','current']].values.tolist()
        interactions = convert_to_interaction(unique_rows_list)
        return interactions

#--------------Spectra-----------------












