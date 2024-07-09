import tomli
import logging as lg
import h5py
import tables
import pandas as pd
import numpy as np
import shutil
import os
import sys
import ranges as rg
from interactions import create_interaction
from datasets import create_dataset
from spectra import create_spectrum
from helper import convert_to_interaction, dset_id_to_name



class Weighting:
    def __init__(self, config_file, log_dir) -> None:
        
        self.pdgid2name = {-16:'anti_tau',-14:'anti_muon',-12:'anti_elec',12:'elec',14:'muon',16:'tau'}

        ##read_config
        lg.info(f'Loading config from {config_file}')
        with open(config_file, "rb") as f:
            self.config = tomli.load(f)#
            self.config['interactions']['unions'] = convert_to_interaction(self.config['interactions']['unions'])
            self.config['misc'] = self.config.get("misc", {})
            #

        ##set paths
        lg.info('Setting paths')
        self.input = self.config['paths']['input']
        self.output = self.config['paths']['output']
        output_dir = os.path.dirname(self.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.logs = log_dir #self.config['paths']['logs']
        
        ##load file
        lg.info(f'Loading file before cuts from {self.input}')
        self.y_df = self._load_df_file_from_dlh5_path(self.input)

        #apply cuts
        lg.info(f'Rebuilding file with cuts')
        self.y_df = self._rebuild_df_with_cuts()

        #create weight_calculations
        lg.info('Initialising dataset')
        dataset_behaviour = self.config['datasets']['behaviour']
        dataset_unions = self.config['datasets']['unions']
        self.dataset = create_dataset(dataset_behaviour, dataset_unions, self.y_df)#TODO: change to initial(self)?

        lg.info('Initialising interaction')
        interaction_behaviour = self.config['interactions']['behaviour']
        interaction_unions = self.config['interactions']['unions']
        self.interaction = create_interaction(interaction_behaviour, interaction_unions, self.y_df)

        lg.info('Initialising spectrum')
        self.spectra = create_spectrum(self.config['spectrum'], self.y_df)

        #calculate weights
        #apply_spectra
        #TODO: Can i add unbalanced functions with creation
        lg.info('Calculating weights for spectra')
        log_path = os.path.join(self.logs, 'spectrum')
        dset_unb = self.dataset.unbalanced_unions
        itype_unb = self.interaction.unbalanced_unions
        
        self.event_weights = np.ones(len(self.y_df))
        self.spectra.log_status(dset_unb, itype_unb, log_path, self.event_weights, 'before')
        self.event_weights = self.spectra.calculate_weights(dset_unb, itype_unb)
        self.spectra.log_status(dset_unb, itype_unb, log_path, self.event_weights, 'after')

        #apply interaction ratios
        lg.info('Calculating weights for interactions')
        log_path = os.path.join(self.logs, 'interaction')
        self.interaction.log_status(log_path, self.event_weights, 'before')
        self.event_weights = self.interaction.calculate_weights(self.event_weights, dset_unb)
        self.interaction.log_status(log_path, self.event_weights, 'after')

        #apply_dataset
        lg.info('Calculating weights for spectra')
        log_path = os.path.join(self.logs, 'dataset')
        self.dataset.log_status(log_path, self.event_weights, 'before')
        self.event_weights = self.dataset.calculate_weights(self.event_weights)
        self.dataset.log_status(log_path, self.event_weights, 'after')

        #change total weight sum
        self.event_weights *= self.config['misc'].get('factor', 1)

        #save weights
        lg.info('Adding sampleweights to file')
        self._add_weightmask_to_dlh5()

    def _load_df_file_from_dlh5_path(self, path):
        columns=(
            "particle_type",
            "is_cc",
            "dataset_id",
            "energy",
            "dir_x",
            "dir_y",
            "dir_z",
            "weight_w2",
            "n_gen",
            "run_id",
            "livetime_s",
            "mc_livetime"
        )

        with h5py.File(path, "r") as f:
            dataset_keys = f["y"].dtype.names
            lg.debug((dataset_keys))
            # Filter out keys not present in the dataset
            valid_columns = tuple([key for key in columns if key in dataset_keys])
            lg.debug(columns)
            lg.debug(type(columns))
            lg.debug(valid_columns)
            lg.debug(type(valid_columns))
            # Log a warning for each removed key
            removed_keys = set(columns) - set(valid_columns)
            for removed_key in removed_keys:
                lg.warning(f"Key '{removed_key}' not present in the dataset. Removed it from columns.")

            # Create the DataFrame with valid columns
            df = pd.DataFrame(f["y"][valid_columns])
            df['dataset'] = df['dataset_id'].map(dset_id_to_name)
            df["current"] = np.where(df["is_cc"] == 2, "CC", "NC")
            df["coszenith"] = -df["dir_z"]
            df["zenith"] = np.arccos(df["coszenith"])
            horizontal_radius = np.sqrt(df["dir_x"]**2+df["dir_y"]**2)
            df["azimuth"] = np.sign(df["dir_y"])*np.arccos(df["dir_x"]/horizontal_radius)


        return df

    def _rebuild_df_with_cuts(self):
        #calculate cuts
        lg.info('Calculating cuts')
        lg.info('Init ranges')
        lg.debug(self.config['ranges'])
        self.ranges = rg.Ranges(self.config['ranges'])#TODO: change to factory?
        lg.info('Add ranges from unions')
        self.ranges.add_ranges_from_datasets(self.config['datasets']['unions'])
        self.ranges.add_ranges_from_interactions(self.config['interactions']['unions'])
        lg.info('Calculate ranges')
        cut_weights = self.ranges.calculate_range_weights(self.y_df)

        ##rebuild with ranges
        self._build_output_file_with_necessary_cuts(cut_weights, self.output)
        lg.info(f'Load output file from {self.output}')

        return self._load_df_file_from_dlh5_path(self.output)

    def _build_output_file_with_necessary_cuts(self, cut, output):
        if np.all(cut == 1):
            lg.info(f'No cuts necessary. Copying file from {self.input} to {output}')
            shutil.copy(self.input, output)

        else:
            lg.info('Applying necessary cuts')
            lg.info(f'Reducing {len(cut)} events to {np.count_nonzero(cut)}')

            with tables.open_file(self.input, "r") as input_f, tables.open_file(output, "w") as output_f:
                self._build_new_x_indices(input_f, output_f, cut)
                self._build_new_x(input_f, output_f, cut)
                self._build_new_y(input_f, output_f, cut)
                self._build_new_group_info(input_f, output_f, cut)

    def _build_new_x(self, input_f, output_f, cut):
        x_mask = self._build_x_mask(cut)
        input_table = input_f.root['x']
        output_table = output_f.create_earray(output_f.root, 'x', atom=tables.Atom.from_dtype(input_table.dtype), shape=(0, input_table.shape[1]))
        self._copy_attributes(input_table, output_table)

        mask_index = 0
        for row in input_table.iterrows():
            if x_mask[mask_index]:
                output_table.append([row[:]])  

            mask_index += 1

    def _build_x_mask(self, y_mask):
        lg.info('Build x_mask')
        with h5py.File(self.input, "r") as f:
            x_indices = pd.DataFrame(f["x_indices"][()])
            reduced_x_indices = x_indices[y_mask]
            x_mask = np.zeros((f["x"].shape[0],),dtype=bool)

            for _, event in reduced_x_indices.iterrows():
                first_event_row = event['index']
                last_event_row = event['index'] + event['n_items']
                x_mask[first_event_row:last_event_row] = True

        return x_mask

    def _build_new_y(self, input_f, output_f, cut):
        input_table = input_f.root['y']
        output_table = output_f.create_table(output_f.root, 'y', description=input_table.description)
        #self._copy_attributes(input_table, output_table)

        mask_index = 0
        for row in input_table.iterrows():
            if cut[mask_index]:
                output_table.append([row[:]])  

            mask_index += 1

    def _build_new_group_info(self, input_f, output_f, cut):
        if not 'group_info' in input_f.root._v_children:
            lg.warn(f'Key ``group_info`` not in {self.input}')
            return
        
        input_table = input_f.root['group_info']
        output_table = output_f.create_table(output_f.root, 'group_info', description=input_table.description)
        #self._copy_attributes(input_table, output_table)

        mask_index = 0
        for row in input_table.iterrows():
            if cut[mask_index]:
                output_table.append([row[:]])  

            mask_index += 1

    def _build_new_x_indices(self, input_f, output_f, cut):
        input_table = input_f.root['x_indices']
        output_table = output_f.create_table(output_f.root, 'x_indices', description=input_table.description)
        #self._copy_attributes(input_table, output_table)

        mask_index = 0
        previous_row = (0,0)
        for row in input_table.iterrows():
            if cut[mask_index]:
                new_row = (sum(previous_row),row[1])
                previous_row = new_row
                output_table.append([new_row])  

            mask_index += 1

    def _copy_attributes(self, src_table, dst_table):
        for attrname in src_table._v_attrs._v_attrnamesuser:
            dst_table.set_attr(attrname, getattr(src_table._v_attrs, attrname))

    def _add_weightmask_to_dlh5(self):
        with h5py.File(self.output, "r+") as f:
            col = "sample_weights"
            y = np.array(f["y"])

            if col in y.dtype.names:
                f["y"][col] = self.event_weights
            else:
                new_dtype = np.dtype(y.dtype.descr + [(col, "<f8")])
                new_ys = np.empty(y.shape, dtype=new_dtype)
                for field in y.dtype.fields:
                    new_ys[field] = y[field]
                new_ys[col] = self.event_weights

                del f["y"]
                f["y"] = new_ys

def conf_to_log(conf_path):
    log_root = '/home/wecapstor3/capn/mppi132h/GNNs/logs/sample_weights/real'

    _, end = conf_path.split('dset_tomls/', 1)
    new_path = os.path.join(log_root, end)
    new_path = new_path.replace('.toml', '/')

    return new_path

def copy_attributes(from_table, new_table):
    for attrname in from_table._v_attrs._v_attrnamesuser:
        new_table.set_attr(attrname, getattr(from_table._v_attrs, attrname))

def prAtr(message, file_obj):
    print(message)
    x_tab = file_obj.root.x
    attrs = x_tab._v_attrs
    print('in_check',attrs)
    # for attrname in attrs._v_attrnamesuser:
    #     if attrname not in attrs._v_unimplemented:
    #         print(attrname, getattr(x_tab._v_attrs, attrname))
    #     else:
    #         print('unimplemented', attrname, getattr(x_tab._v_attrs, attrname))


if __name__ == '__main__':

    conf_path = sys.argv[1]
    log_path = conf_to_log(conf_path)
    # Configure logging to write to a file
    lg.basicConfig(
        level=lg.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        filename=f'{log_path}/info.log',
        encoding='utf-8'
    )

    lg.info('Start')
    Weighting(config_file=conf_path, log_dir= log_path)