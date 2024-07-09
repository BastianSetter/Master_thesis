from km3services.oscprob import OscProb
from km3flux.flux import Honda
import numpy as np
import pandas as pd
from helper import get_mask_for_du_ptype_cc_on_df, Interaction, _unique_interactions_in_df


def physical_honda_flux(df:pd.DataFrame):
    '''
    df: dataframe containing the data for a unbalanced dataset
    '''
    physical_flux = np.ones(shape=(len(df),))
    honda_flux = Honda().flux(year=2014, experiment="Frejus", averaged=None)
    oscprob = OscProb()

    unique_interactions = _unique_interactions_in_df(df)
    interaction:Interaction
    for interaction in unique_interactions:
        pdgid = interaction.pdgid
        interaction_mask = get_mask_for_du_ptype_cc_on_df(df, interaction=interaction)
        df_interaction = df[interaction_mask]

        isAnti = True if pdgid < 0 else False
        energies = df_interaction["energy"]
        coszeniths = df_interaction["coszenith"]
        azimuths = df_interaction["azimuth"]

        elec_flux_str = "anue" if isAnti else "nue"
        muon_flux_str = "anumu" if isAnti else "numu"
        elec_flux = honda_flux[elec_flux_str](energies, coszeniths, azimuths)
        muon_flux = honda_flux[muon_flux_str](energies, coszeniths, azimuths)

        pdgid_e = -12 if isAnti else 12
        pdgid_mu = -14 if isAnti else 14
        oscprob_e = oscprob.oscillationprobabilities(
            pdgid_e, pdgid, energies, coszeniths
        )
        oscprob_mu = oscprob.oscillationprobabilities(
            pdgid_mu, pdgid, energies, coszeniths
        )

        physical_flux[interaction_mask] = (oscprob_e * elec_flux + oscprob_mu * muon_flux)
    return physical_flux

def calculate_physical_neutrino_weights_for_df(df:pd.DataFrame) -> np.ndarray:
    physical_flux = physical_honda_flux(df)
    weights_w2 = df['weight_w2']
    n_gen = df['n_gen']
    livetime = df['livetime_s']

    physical_weights = physical_flux*weights_w2*livetime/n_gen
    return physical_weights

def calculate_physical_muon_weights_for_df(df:pd.DataFrame) -> np.ndarray:
    livetime = df['livetime_s']
    mc_livetime = df['mc_livetime']

    physical_weights = livetime/mc_livetime
    return physical_weights

def calculate_physical_weights_for_df(df:pd.DataFrame) -> np.ndarray:
    physical_weights = np.ones(shape=(len(df),))
    neutrino_mask = df['particle_type'].isin([12, 14, 16, -12, -14, -16])
    if np.sum(neutrino_mask):
        physical_weights[neutrino_mask] = calculate_physical_neutrino_weights_for_df(df[neutrino_mask])
    muon_mask = df['particle_type'].isin([13, -13])
    if np.sum(muon_mask):
        physical_weights[muon_mask] = calculate_physical_muon_weights_for_df(df[muon_mask])

    normalised_physical_weights = physical_weights/sum(physical_weights)*len(physical_weights)
    return np.array(normalised_physical_weights)