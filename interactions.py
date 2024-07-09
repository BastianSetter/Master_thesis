import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from physical_model import calculate_physical_weights_for_df
import os
from helper import remove_suffix, recursive_sum, itype_tree_to_str, get_mask_for_du_ptype_cc_on_df, _extend_unions, convert_to_interaction
import matplotlib.pyplot as plt



class BaseInteraction(ABC):
    def __init__(self, input, unions, df) -> None:#DONE
        unions = _extend_unions(unions)
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
        self.sub_interactions = []
        for sub_behaviour, sub_union in zip(input[1:], unions):
            child = create_interaction(sub_behaviour, sub_union, self.df)
            self.sub_interactions.append(child)

        sub_interaction:BaseInteraction
        for counter, sub_interaction in enumerate(self.sub_interactions):
            mask = sub_interaction._get_managed_events()
            id_to_assign = counter+1
            self.union_id[mask] = id_to_assign
            self.ids.append(id_to_assign)

    def _get_managed_events(self):#TODO: property
        return self.union_id != 0
    
    def _get_unique_managed_ids(self, union_id = None):#TODO: Einfacher durch die nutzung von self.ids?
        if union_id is None:
            union_id = self.union_id
        all_ids = np.unique(union_id)
        ids = all_ids[all_ids != 0]
        return ids


    def _init_as_child(self, unions):
        self.itypes = unions
        for counter, interaction in enumerate(self.itypes):
            mask = get_mask_for_du_ptype_cc_on_df(self.df, interaction=interaction)
            id_to_assign = counter+1
            self.union_id[mask] = id_to_assign
            self.ids.append(id_to_assign)

    def calculate_weights(self, current_weights, unbalanced_dataset_func):
        if hasattr(self, 'sub_interactions'):
            sub_interaction:BaseInteraction
            for sub_interaction in self.sub_interactions:
                single_sub_weights = sub_interaction.calculate_weights(current_weights, unbalanced_dataset_func)
                sub_mask = sub_interaction._get_managed_events()
                current_weights[sub_mask] = single_sub_weights

        
        layer_weights = self._calculate_weights_in_layer_for_all_datasets(current_weights, unbalanced_dataset_func)
        
        mask = self._get_managed_events()
        assert len(layer_weights) == np.count_nonzero(mask)
        assert np.isclose(np.sum(layer_weights), np.sum(current_weights[mask]))

        return layer_weights

    def _calculate_weights_in_layer_for_all_datasets(self, current_weights:np.ndarray, unbalanced_dataset_func):
        layer_weights = current_weights.copy()
        unbalanced_datasets = unbalanced_dataset_func()
        class_event_mask = self._get_managed_events()
        for unbalanced_set in unbalanced_datasets:
            dset_mask = get_mask_for_du_ptype_cc_on_df(self.df, dataset_union=unbalanced_set)
            total_mask = np.logical_and(dset_mask, class_event_mask)
            single_dset_weights = self._calculate_weights_in_layer(current_weights, total_mask)
            
            assert len(single_dset_weights) == np.count_nonzero(total_mask)
            assert np.isclose(np.sum(single_dset_weights), np.sum(current_weights[total_mask]))
            layer_weights[total_mask] = single_dset_weights

        return layer_weights[class_event_mask]


    def _check_inner_lists(self, input_list):
        for child_list in input_list:
            if len(child_list) != 1:
                    return False
        return True

    def unbalanced_unions(self):
        if hasattr(self, 'sub_interactions'):
            child_interactions = [child.unbalanced_unions() for child in self.sub_interactions]
            all_children_unbalanced = self._check_inner_lists(child_interactions)
            if self._unbalanced and all_children_unbalanced:
                return [child_list[0] for child_list in child_interactions]
                    
            else:
                return [unbalanced_part for child_list in child_interactions for unbalanced_part in child_list]
            
        else:
            if self._unbalanced:
                return [self.itypes]
            else:
                seperated_itypes = [[itype] for itype in self.itypes]
                return seperated_itypes

    def log_status(self, path, current_weights, file_add = ''):
        if not os.path.exists(path):
            os.makedirs(path)
        unique_ids = self._get_unique_managed_ids()
        contained_sets = []
        total_weights = []

        if hasattr(self, 'sub_interactions'):
            sub_interaction:BaseInteraction
            for id, sub_interaction in zip(unique_ids, self.sub_interactions):
                class_name = sub_interaction.__class__.__name__
                short_name = remove_suffix(class_name, "Interaction")
                contained_set, total_weight = sub_interaction.log_status(os.path.join(path,f'{short_name}_id_{id}'), current_weights, file_add)
                contained_sets.append(contained_set)
                total_weights.append(total_weight)
            
        else:
            for id in unique_ids:
                mask = self.union_id == id
                single_total_weight = np.sum(current_weights[mask])
                total_weights.append(single_total_weight)

                id_set_names = self.df.loc[mask, ('particle_type','current')].drop_duplicates()
                assert len(id_set_names) == 1
                contained_sets.append(convert_to_interaction(id_set_names.iloc[0]))

        self._plot_histogram(contained_sets, total_weights, path, file_add)

        return contained_sets, total_weights
    
    def _plot_histogram(self, contained_sets, total_weights, path, file_add):
        
        unique_ids = self._get_unique_managed_ids()
        plt.figure(figsize=(5,3))
        for counter, (union_weight, union_sets, id) in enumerate(zip(total_weights, contained_sets, unique_ids)):
            weight_sum = recursive_sum([union_weight])
            label = f'id_{id}: {weight_sum}: ' + itype_tree_to_str([union_sets])
            plt.bar(counter, weight_sum, label = label)
        
        plt.xlabel('Interaction unions')
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
    def _calculate_weights_in_layer(self, current_weights, mask):
        ...

class AsIsInteraction(BaseInteraction):
    _unbalanced = True
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    def _calculate_weights_in_layer(self, current_weights, mask):
        return current_weights[mask]
    

class TargetInteraction(BaseInteraction):
    _unbalanced = False
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    def _weights_from_target(self, current_weights:np.ndarray, targets, mask):
        current_weights_red, union_id_red = current_weights[mask], self.union_id[mask]
        unique_union_ids = self._get_unique_managed_ids(union_id_red)

        total_weights = []
        for id in unique_union_ids:
            id_mask = union_id_red == id
            single_total_weight = np.sum(current_weights_red[id_mask])
            total_weights.append(single_total_weight)
        total_weights = np.array(total_weights)

        total_weight_sum = np.sum(total_weights)
        total_target_sum = np.sum(targets)
        scaling_factor = targets / total_weights * total_weight_sum / total_target_sum# hier ändern

        union_weights = current_weights_red.copy()
        for id, weight_shift in zip(unique_union_ids, scaling_factor):
            id_mask = union_id_red == id# muss häufig behalten werden
            union_weights[id_mask] *= weight_shift

        return union_weights


class PhysicalFlatInteraction(TargetInteraction):
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    def _calculate_physical_weights_in_layer(self, current_weights, mask):
        union_id_red = self.union_id[mask]
        unique_union_ids = self._get_unique_managed_ids(union_id_red)   
        physical_weights = calculate_physical_weights_for_df(self.df[mask])

        targets = []
        for id in unique_union_ids:
            id_mask = union_id_red == id
            total_physical_weight = np.sum(physical_weights[id_mask])
            targets.append(total_physical_weight)
        targets = np.array(targets)
        
        return self._weights_from_target(current_weights, targets, mask)
    
    def _calculate_flat_weights_in_layer(self, current_weights, mask):
        union_id_red = self.union_id[mask]
        unique_union_ids = self._get_unique_managed_ids(union_id_red)   

        targets = np.ones((len(unique_union_ids),))
        
        return self._weights_from_target(current_weights, targets, mask)
    
    def _calculate_weights_in_layer(self, current_weights, mask):
        flat_weights = self._calculate_flat_weights_in_layer(current_weights, mask)
        physical_weights = self._calculate_physical_weights_in_layer(current_weights, mask)

        ratio = self.type_input[0]
        ratio_weights = physical_weights * ratio + flat_weights * (1-ratio)

        return ratio_weights


class FlatInteraction(PhysicalFlatInteraction):
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    def _calculate_weights_in_layer(self, current_weights, mask):
        return self._calculate_flat_weights_in_layer(current_weights, mask)

class PhysicalInteraction(PhysicalFlatInteraction):
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)

    def _calculate_weights_in_layer(self, current_weights, mask):
        return self._calculate_physical_weights_in_layer(current_weights, mask)


class CustomInteraction(TargetInteraction):
    def __init__(self, input, unions, df) -> None:
        super().__init__(input, unions, df)
        assert len(input[0]) == len(input[1:])

    def _calculate_weights_in_layer(self, current_weights, mask):
        targets = np.array(self.type_input)

        return self._weights_from_target(current_weights, targets, mask)
        



def create_interaction(behaviour, unions, df) -> BaseInteraction:
    if len(behaviour)>0:
        type = behaviour[0]
    else:
        type = 'AsIs'

    input = behaviour[1:]

    return type_class_match[type](input, unions, df)

type_class_match = {
    'AsIs': AsIsInteraction,
    'PhysicalBalancedRatio': PhysicalFlatInteraction,
    'Custom': CustomInteraction,
    'Balanced': FlatInteraction,
    'Physical': PhysicalInteraction
}

if __name__ == '__main__':
    siz = 100
    t_unions = [
        [[12, "CC"],[-12, "CC"]],
        [[14, "CC"],[-14, "CC"]],
        [[14, "NC"],[-14, "NC"]],
        [[16, "CC"],[-16, "CC"]]
    ]
    t_unions = convert_to_interaction(t_unions)

    t_behaviour = ['Balanced',[],
        ["Physical",[]],
        ["Physical",[]],
        ["Physical",[]],
        ["Physical",[]]
    ]

    test_weights = np.random.uniform(0.1, 10, (siz))

    dset_name_to_id = {
        'lukas_flat': 0,
        'standard_v7.1': 7.1,
        'standard_v7.2': 7.2
    }

    

    values = [[12, "CC"], [-12, "CC"], [14, "CC"], [-14, "CC"], [14, "NC"], [-14, "NC"], [16, "CC"], [-16, "CC"]]
    indices = np.random.randint(len(values), size=(siz))
    test_dset_itype = np.array(values)[indices]

    t_df = pd.DataFrame(test_dset_itype, columns=['particle_type', 'current'])

    dset_values = list(dset_name_to_id.values())
    test_dset_id = np.random.choice(dset_values, size=(siz))

    t_df['dataset'] = test_dset_id

    for col_name in ['n_gen', 'weights_w2', 'livetime_s', 'energy', 'coszenith', 'azimuth']:
        test_data_col = np.random.uniform(0.1, 1, (siz))
        t_df[col_name] = test_data_col

    def unbalanced():
        t_unions = [[0], [7.1, 7.2]]
        return t_unions

    t_set = create_interaction(t_behaviour, t_unions, t_df)
    print('----------------------------------------')
    t_set.calculate_weights(test_weights, unbalanced)