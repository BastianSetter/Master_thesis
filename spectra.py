import numpy as np
from itertools import product
from abc import ABC, abstractmethod
from helper import itype_tree_to_str, dset_tree_to_str, get_mask_for_du_ptype_cc_on_df
from physical_model import calculate_physical_weights_for_df
import os
import matplotlib.pyplot as plt
import logging
lg = logging.getLogger(__name__)




class BaseSpectrum(ABC):
    def __init__(self, config, df) -> None:
        self.config = config
        self.df = df
    
    def calculate_weights(self, unbalanced_dset_func, unbalanced_itype_func):
        unbalanced_dsets = unbalanced_dset_func()
        unbalanced_itype = unbalanced_itype_func()
        unbalanced_combinations = product(unbalanced_dsets, unbalanced_itype)

        event_weights = np.ones(len(self.df))
        for dset, itype in unbalanced_combinations:
            lg.info(f'Handling {dset} and {itype}')
            mask = get_mask_for_du_ptype_cc_on_df(self.df,dataset_union=dset,interaction=itype)
            masked_df = self.df[mask]
            partial_weights = self._calculate_weights_unbalanced_part(masked_df)
            event_weights[mask] = partial_weights
        
        return event_weights


    @abstractmethod
    def _calculate_weights_unbalanced_part(self, df):
        ...

    def log_status(self, unbalanced_dset_func, unbalanced_itype_func, path, current_weights, file_add = ''):
        unbalanced_dsets = unbalanced_dset_func()
        unbalanced_itype = unbalanced_itype_func()
        unbalanced_combinations = product(unbalanced_dsets, unbalanced_itype)
        for (dset, itype) in unbalanced_combinations:
            dset_path = dset_tree_to_str(dset)
            itype_path = itype_tree_to_str(itype)
            dst_path = os.path.join(path, dset_path, itype_path)

            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            mask = get_mask_for_du_ptype_cc_on_df(self.df,dataset_union=dset,interaction=itype)
            self._log_status_for_combination(dst_path, current_weights[mask], self.df[mask], file_add)

    def _log_status_for_combination(self, path, weights, df, file_add):
        title = None if file_add == '' else file_add
        file_name = f'spectrum_{file_add}'
        save_path = os.path.join(path,file_name)
#        metrics = self.config[metrics] if 'columns' in self.config.keys() else None
        self._plot_weighted_histograms(title, save_path, df, weights)#, metrics)

    def _plot_weighted_histograms(self, title, save, masked_df, weights, metrics=None):
        if metrics is None:
            metrics = [("energy", 50), ("azimuth", 20), ("coszenith", 20)]

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        if title:
            fig.suptitle(title, y=1.01)
        fig.subplots_adjust(wspace=0.4, hspace=0.5)

        for counter, (metric, bins) in enumerate(metrics):
            ax1 = axes[0][counter]
            
            # Get the previous metric and bins for comparison
            prev_metric, prev_bins = metrics[counter - 1]
            
            # Create a 2D histogram
            hist2d = ax1.hist2d(
                masked_df[metric],
                masked_df[prev_metric],
                bins=[bins, prev_bins],
                weights=weights,
                cmap=plt.cm.jet
            )
            plt.colorbar(hist2d[3], ax=ax1, location='top', label='counts')
            ax1.set_xlabel(metric)
            ax1.set_ylabel(prev_metric)

            ax2 = axes[1][counter]
            
            # Create a 1D histogram
            ax2.hist(
                masked_df[metric],
                bins=bins,
                weights=weights,
                color="royalblue"
            )
            ax2.set_xlabel(metric)

        axes[1][0].set_ylabel('counts')

        # Save the figure and close
        for format in ['pdf','png']:
            plt.savefig(f'{save}.{format}', bbox_inches = 'tight')
        plt.close()

class EnergyCoszenithAzimuthPhysicalFlat(BaseSpectrum):
    def __init__(self, config, df) -> None:
        super().__init__(config, df)

    def _calculate_weights_unbalanced_part(self, df):
        physical_weights = calculate_physical_weights_for_df(df)
        lg.debug(f'physical weights {sum(physical_weights)}')
        flat_weights = self._calculate_flat_weights_for_df(df)
        lg.debug(f'flat weights {sum(flat_weights)}')

        ratio = float(self.config['ratio'])

        ratio_weights = physical_weights * ratio + flat_weights * (1-ratio)

        return ratio_weights
    
    def _flat_bin_scaling_factors_from_data(self, data_values:np.ndarray, num_bins, weights=None, optimize_bins = True):
        if optimize_bins:
            # Set the maximum number of iterations for bin adjustment
            max_iterations = 100
            
            # Iterate to adjust bin counts iteratively for achieving minimum bin count of 5
            for iteration in range(max_iterations):
                # Calculate the histogram with weighted data using np.histogramdd
                try:
                    hist, bin_edges = np.histogramdd(data_values, bins=num_bins, weights=weights)
                except TypeError as te:
                    print(num_bins)
                    raise te
                
                MINIMUM_BIN_AMOUNT = 3

                # Identify bins with low counts (less than 5) and adjust binning
                low_count_bins = np.argwhere(hist < MINIMUM_BIN_AMOUNT)
                if len(low_count_bins) == 0:
                    lg.debug(f'Optimisation converged after {iteration} iterations.')
                    break  # No low count bins, exit loop
                
                # Store the current bin counts to detect convergence
                old_num_bins = num_bins[:]
                
                # Reduce bin counts for each dimension
                for idx, dim in enumerate(num_bins):
                    num_bins[idx] = int(np.round(dim / 1.05))  # Reduce bin count 
                
                # If bin counts didn't change, perform a faster reduction
                if old_num_bins == num_bins:
                    for idx, dim in enumerate(num_bins):
                        num_bins[idx] = int(dim // 1.05)
        
        # Calculate total weight if weights are provided, else use number of data points
        total_weight = np.sum(weights) if weights is not None else data_values.shape[0]
        lg.debug(f'Use bins: {num_bins}; total is {np.prod(num_bins)}')
        # Calculate the target weight per bin
        target_bin_weight = total_weight / np.prod(num_bins)

        # Recalculate the histogram with adjusted binning
        hist, bin_edges = np.histogramdd(data_values, bins=num_bins, weights=weights)
        
        # Calculate the scaling factor for each bin
        scaling_factors = target_bin_weight / hist
        
        # Digitize the data to get bin indices for each data point
        bin_indices = [np.digitize(data_values[:, i], bin_edges[i]) - 1 for i in range(data_values.shape[1])]
        
        # Clip bin_indices to ensure they are within valid range
        for i in range(data_values.shape[1]):
            bin_indices[i] = np.clip(bin_indices[i], 0, num_bins[i] - 1)
        
        # Get the scaling factor for each data point based on its bin index
        scaling_factors_for_data = scaling_factors[tuple(bin_indices[i] for i in range(data_values.shape[1]))]

        return scaling_factors_for_data

    def _calculate_flat_weights_for_df(self, df):
        if 'bins' in self.config.keys():
            bins = self.config['bins']
        else:
            bins = [50,20,20]

        if 'columns' in self.config.keys():
            columns = self.config['columns']
        else:
            columns = ['energy', 'coszenith', 'azimuth']
        
        flat_weights = np.ones(len(df))
        for column in columns:
            col_flat_weight_change = self._flat_weights_1d(df.copy(), column)
            flat_weights *= col_flat_weight_change
            
        if len(columns) > 1:
            columns_str = '-'.join(columns)
            lg.info(f'Flattening combined {columns_str}')

            select_columns_data = df[columns].values
            flat_weights_change = self._flat_bin_scaling_factors_from_data(select_columns_data, bins, weights = flat_weights,optimize_bins=True)
            flat_weights *= flat_weights_change

        return flat_weights

    def _flat_weights_1d(self, df, column):
        lg.info(f'Flattening {column}')
        # Calculate the minimum number of events per bin (at least 50)

        min_events_per_bin = max(50, len(df) // 1000)
        lg.info(f'Splitting {len(df)} events into chunks of size {min_events_per_bin}, resulting in {len(df)// min_events_per_bin} chunks.')
        # Step 1: Calculate the bin edges to achieve the desired events per bin
        bin_edges = np.percentile(df[column], np.linspace(0, 100, num=len(df) // min_events_per_bin + 1))

        # Step 2: Assign weights to each event based on bin width
        df['weights'] = 0.0
        for i in range(len(bin_edges) - 1):
            mask = (df[column] >= bin_edges[i]) & (df[column] < bin_edges[i + 1])
            bin_width = bin_edges[i + 1] - bin_edges[i]
            lg.debug(f'Chunk from {bin_edges[i]} to {bin_edges[i+1]} with width {bin_width}')
            df.loc[mask, 'weights'] = bin_width

        df.loc[df[column].idxmax(), 'weights'] = bin_edges[-1] - bin_edges[-2]

        # Step 3: Normalize the weights
        total_weight = df['weights'].sum()
        df['weights'] = df['weights'] / total_weight * len(df)

        
        return df['weights'].values


class EnergyCoszenithAzimuthFlat(EnergyCoszenithAzimuthPhysicalFlat):
    def __init__(self, config, df) -> None:
        super().__init__(config, df)

    def _calculate_weights_unbalanced_part(self, df):
        return self._calculate_flat_weights_for_df(df)
    
class EnergyCoszenithAzimuthPhysical(EnergyCoszenithAzimuthPhysicalFlat):
    def __init__(self, config, df) -> None:
        super().__init__(config, df)

    def _calculate_weights_unbalanced_part(self, df):
        return calculate_physical_weights_for_df(df)


def create_spectrum(config, df) -> BaseSpectrum:
    type = config['type']
    return type_class_match[type](config, df)

type_class_match = {
    'PhysicalBalancedRatio': EnergyCoszenithAzimuthPhysicalFlat,
    'Balanced': EnergyCoszenithAzimuthFlat,
    'Physical': EnergyCoszenithAzimuthPhysical
}