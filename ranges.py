import numpy as np
import logging 
lg = logging.getLogger(__name__)
import pandas as pd
from helper import get_mask_for_du_ptype_cc_on_df, get_all_datasets_from_all_unions, get_all_interactions_from_all_unions

class Ranges():
    def __init__(self, config) -> None:
        self.config = config
        
    def add_ranges_from_datasets(self, dset_unions):
        self.config["sets"] = get_all_datasets_from_all_unions(dset_unions)

    def add_ranges_from_interactions(self, itype_unions):
        self.config["interactions"] = get_all_interactions_from_all_unions(itype_unions)

    def calculate_range_weights(self, y_df:pd.DataFrame):
        cut_weights = np.ones(shape=len(y_df),dtype=bool)
        for range in self.config["ranges"]:
            
            type, start, stop = range
            valid_data_points = (y_df[type]>start) & (y_df[type]<stop)
            
            lg.debug(f'{type}; {y_df[type].min()}; {y_df[type].max()}; {start}; {stop}')
            cut_weights &= np.array(valid_data_points)
            
        dsets = self.config["sets"]
        interactions = self.config["interactions"]
        category_mask = get_mask_for_du_ptype_cc_on_df(y_df, dataset_union=dsets, interaction=interactions)
        cut_weights &= category_mask

        return np.array(cut_weights)

