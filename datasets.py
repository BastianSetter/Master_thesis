import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
import os
from helper import remove_suffix, recursive_sum, dset_tree_to_str, get_mask_for_du_ptype_cc_on_df, dset_name_to_id
import matplotlib.pyplot as plt




class BaseDataset(ABC):
    def __init__(self, input, unions, df) -> None:
        self.df = df
        self.union_id = np.zeros((len(self.df),))
        self.ids = []

        if len(input)>0:
            self.type_input = input[0]
        else:
            self.type_input = []
        
        if len(input) > 1:#TODO: Check if sufficient as check for lowest level
            self._init_as_parent(input, unions)
        else:
            self._init_as_child(unions)

    def _init_as_parent(self, input, unions):
        assert len(unions) == len(input[1:])
        self.sub_datasets = []
        for subbehaviour, subunion in zip(input[1:], unions):
            child = create_dataset(subbehaviour, subunion, self.df)
            self.sub_datasets.append(child)

        sub_dataset:BaseDataset
        for counter, sub_dataset in enumerate(self.sub_datasets):
            mask = sub_dataset._get_managed_events()
            id_to_assign = counter+1
            self.union_id[mask] = id_to_assign
            self.ids.append(id_to_assign)

    def _get_managed_events(self):
        return self.union_id != 0
    
    def _get_unique_managed_ids(self):
        all_ids = np.unique(self.union_id)
        ids = all_ids[all_ids != 0]
        return ids

    def _init_as_child(self, unions):
        self.dset_names = unions
        
        for counter, dset_name in enumerate(unions):
            mask = get_mask_for_du_ptype_cc_on_df(self.df, dataset_union = dset_name)
            id_to_assign = counter+1
            self.union_id[mask] = id_to_assign
            self.ids.append(id_to_assign)

    def calculate_weights(self, current_weights):
        if hasattr(self, 'sub_datasets'):
            sub_dataset:BaseDataset
            for sub_dataset in self.sub_datasets:
                single_sub_weights = sub_dataset.calculate_weights(current_weights)
                sub_mask = sub_dataset._get_managed_events()
                current_weights[sub_mask] = single_sub_weights

        layer_weights = self._calculate_weights_in_layer(current_weights)

        mask = self._get_managed_events()        
        assert len(layer_weights) == np.count_nonzero(mask)
        assert np.isclose(np.sum(layer_weights), np.sum(current_weights[mask]))

        return layer_weights

    def _check_inner_lists(self, input_list):
        for child_list in input_list:
            if len(child_list) != 1:
                    return False
        return True

    def unbalanced_unions(self):
        if hasattr(self, 'sub_datasets'):
            child_sets = [child.unbalanced_unions() for child in self.sub_datasets]
            all_children_unbalanced = self._check_inner_lists(child_sets)
            if self._unbalanced and all_children_unbalanced:
                return [child_list[0] for child_list in child_sets]
                    
            else:
                return [unbalanced_part for child_list in child_sets for unbalanced_part in child_list]
            
        else:
            if self._unbalanced:
                return [self.dset_names]
            else:
                seperated_sets = [[dset_name] for dset_name in self.dset_names]
                return seperated_sets

    def log_status(self, path, current_weights, file_add = ''):
        if not os.path.exists(path):
            os.makedirs(path)
        unique_ids = self._get_unique_managed_ids()
        contained_sets = []
        total_weights = []

        if hasattr(self, 'sub_datasets'):
            sub_dataset:BaseDataset
            for id, sub_dataset in zip(unique_ids, self.sub_datasets):
                class_name = sub_dataset.__class__.__name__
                short_name = remove_suffix(class_name, "Dataset")
                contained_set, total_weight = sub_dataset.log_status(os.path.join(path,f'{short_name}_id_{id}'), current_weights, file_add)
                contained_sets.append(contained_set)
                total_weights.append(total_weight)
            
        else:
            for id in unique_ids:
                mask = self.union_id == id
                single_total_weight = np.sum(current_weights[mask])
                total_weights.append(single_total_weight)

                dset_names = self.df.loc[mask, 'dataset'].drop_duplicates()
                assert len(dset_names) == 1
                contained_sets.append(dset_names.iloc[0])

        self._plot_histogram(contained_sets, total_weights, path, file_add)

        return contained_sets, total_weights

    def _plot_histogram(self, contained_sets, total_weights, path, file_add):
        
        unique_ids = self._get_unique_managed_ids()
        plt.figure(figsize=(5,3))
        for counter, (union_weight, union_sets, id) in enumerate(zip(total_weights, contained_sets, unique_ids)):
            weight_sum = recursive_sum([union_weight])
            label = f'id_{id}: {weight_sum}: ' + dset_tree_to_str([union_sets])
            plt.bar(counter, weight_sum, label = label)
        
        plt.xlabel('Dataset unions')
        class_name = self.__class__.__name__
        title = f'{class_name}: {file_add}'
        plt.title(title)
        plt.legend(loc = 'upper center', bbox_to_anchor = (0.5,-0.2))
        filename = f'{class_name}_{file_add}.'
        for format in ['pdf','png']:
            plt.savefig(os.path.join(path, filename+format), bbox_inches = 'tight')
        plt.close()

    @property
    @abstractmethod
    def _unbalanced(self) -> bool:
        ...

    @abstractmethod
    def _calculate_weights_in_layer(self, current_weights):
        ...
    

class TargetDataset(BaseDataset):
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    @property
    @abstractmethod
    def _unbalanced(self) -> bool:
        ...

    @abstractmethod
    def _calculate_weights_in_layer(self, current_weights):
        ...

    def _weights_from_target(self, current_weights:np.ndarray, targets):
        unique_union_ids = self._get_unique_managed_ids()

        total_weights = []
        for id in unique_union_ids:
            mask = self.union_id == id
            single_total_weight = np.sum(current_weights[mask])
            total_weights.append(single_total_weight)
        total_weights = np.array(total_weights)

        total_weights_sum = np.sum(total_weights)
        targets_sum = np.sum(targets)
        scaling_factor = targets / total_weights * total_weights_sum / targets_sum# hier ändern

        union_weights = current_weights.copy()#copy
        for id, weight_shift in zip(unique_union_ids, scaling_factor):
            mask = self.union_id == id
            union_weights[mask] *= weight_shift
        return_weights = union_weights[self._get_managed_events()]
        return return_weights


class AsIsDataset(BaseDataset):
    _unbalanced = True
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    def _calculate_weights_in_layer(self, current_weights):
        mask = self._get_managed_events()
        return current_weights[mask] 
    

class BalancedWeightDataset(TargetDataset):
    _unbalanced = False
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    def _calc_total_weights(self, current_weights):
        unique_union_ids = self._get_unique_managed_ids()
        total_weights = []
        for id in unique_union_ids:
            mask = self.union_id == id
            single_total_weight = np.sum(current_weights[mask])
            total_weights.append(single_total_weight)
        return np.array(total_weights)

    def _calculate_weights_in_layer(self, current_weights):
        unique_union_ids = self._get_unique_managed_ids()
        
        targets = np.ones_like(unique_union_ids)

        return self._weights_from_target(current_weights, targets)


class BalancedEventNDataset(TargetDataset):
    _unbalanced = False
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    def _calculate_weights_in_layer(self, current_weights):
        unique_union_ids = self._get_unique_managed_ids()

        targets = []
        for id in unique_union_ids:
            mask = self.union_id == id
            single_n_events = np.count_nonzero(mask)
            targets.append(single_n_events)
        targets = np.array(targets)
        
        return self._weights_from_target(current_weights, targets)


class CustomWeightDataset(TargetDataset):
    _unbalanced = False
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)
        assert len(input[0]) == len(input[1:])

    def _calculate_weights_in_layer(self, current_weights):
        targets = np.array(self.type_input)

        return self._weights_from_target(current_weights, targets)


class AutoRatio(TargetDataset):
    _unbalanced = False
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)
        assert len(input[1:]) == 2
        self.ratio = self.type_input[0]

    def _calculate_weights_in_layer(self, current_weights):
        targets = np.array([self.ratio, 1-self.ratio])
        
        return self._weights_from_target(current_weights, targets)



def create_dataset(behaviour, unions, df) -> BaseDataset:
    if len(behaviour)>0:
        type = behaviour[0]
    else:
        type = 'AsIs'

    input = behaviour[1:]

    return type_class_match[type](input, unions, df)

type_class_match = {
    'AsIs': AsIsDataset,
    'Balanced weight': BalancedWeightDataset,
    'Balanced eventN': BalancedEventNDataset,
    'Custom': CustomWeightDataset,
    'Auto': AutoRatio
}




if __name__ == '__main__':
    

    t_unions = [
        ['lukas_flat'],
        ['standard_v7.2','standard_v7.1']
    ]

    t_behaviour = ['AsIs',[],
        [],
        ['Balanced weight',[]]
    ]

    test_weights = np.random.uniform(0.1,10,(50))

    values = list(dset_name_to_id.values())
    test_dset_id = np.random.choice(values, size=(50))
    t_df = pd.DataFrame(test_dset_id, columns=['dataset'])

    t_set = create_dataset(t_behaviour, t_unions, t_df)
    t_set.calculate_weights(test_weights)
    print(t_set.unbalanced_unions())
