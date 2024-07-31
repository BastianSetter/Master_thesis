# Master thesis: Optimisation of energy reconstruction with sample weights for GNNs in KM3NeT/ORCA6
## Content
This repository contains my master thesis as well as the code to calculate the dataset weights. The code is not functional, as it uses several KM3NeT internal dependencies and some datatables, which are not publicly available. A summary of the thesis content is given later under **Abstract**
## Code
 - The settings for the final weighting of a dataset are provided to the script by a toml file `test_input.toml`
 - This is used to first copy the relevant events (rows) of the original dataset (including all available events) into the output file and then calculate the desired weights.
 - The copying process checks, which data `ranges` are given by the toml and cuts all other. Additionaly only events with a dataset and interaction type tag present int the toml are copied.
 - The calculation of the weights is performed in three steps, after each the weights are normalised so that the dataset, interaction type and spectrum weights can be set independently.
 - As datasets and interaction types both are classification tags, they follow a similar structure. The user can group events together in a arbitrarily deep structure and apply a differnt weighting at each step (again independence assured by normilisation). The ratios for datasets can be set only in limited ways (custom ratios like 1:3), while the interaction type also have the option to switch continuóusly between a flat (equal) and physical (as theory suggests) weighting.
 - The switch between flat and physical weighting as also possible for the spectrum weights including all continous parameters of the datasamples saved in the file. This is typically the energy and direction (coszenith, azimuth).
 - As the script is designed in a modular way, addition of new only requires the implementation of the actual weight calculation in a new class, as every thing else is handled by the abstract parent.
 - The entry point the script is at 'submit_add_weights.py' and first goes thorugh to other files, necessary due to the server infrastructure.

## Abstract
With the discovery of neutrino oscillations by Super-kamiokande and SNO it has
been shown that neutrinos have mass. This conflicts with the current formulation of
the Standard Model of particle physics, where neutrinos are assumed massless. This
makes neutrinos an excellent candidate to explore multiple topics in physics beyond
the Standard Model, like understanding neutrino oscillations, determining the neutrino
mass hierarchy, probing Lorentz invariance or measuring Quantum decoherence.

For the detection of neutrinos, the KM3NeT collaboration is currently building two
water Cherenkov telescopes named ORCA and ARCA. With a spacing of 10 m to 20 m
between its detection units KM3NeT/ORCA excels at the detection of atmospheric
neutrinos in the GeV range. Since KM3NeT/ORCA is still under construction, this
thesis uses data for only 6 of the planned 115 detection units.

All of the just mentioned use cases of atmospheric neutrinos require an accurate energy
reconstruction to give significant results. The best energy reconstructions in KM3NeT
can at the moment be achieved with Graph Neural Networks (GNN).

This thesis tests the effects and possible benefits of applying sample weights in the
training of GNNs. Therefore, three different options to calculate the weights are
evaluated. The best-found configuration is then compared to the likelihood-based
reconstruction algorithms JShower and JMuon, as well as a previous GNN energy
reconstruction by Daniel Guderian. The 3 test sample weight options are the ratio of
standard run-by-run simulations to additional single-run simulations by Lukas Hennig,
the ratio of physical to flat weights for interaction types, and the ratio of physical to
flat weights for the spectrum of energy and arrival direction.
