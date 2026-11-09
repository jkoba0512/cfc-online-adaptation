# Readout adaptation in simulated arm reaching

Code for **Readout adaptation in simulated arm reaching: learned compensation and performance after updates stop**, by Jun Kobayashi.

This manuscript release compares six adaptive variants: dense-backbone CfC LMS, NCP-wired CfC LMS, FC-wired CfC LMS, ESN LMS, RBF LMS, and dense-CfC RLS. All three CfC configurations use final 400-epoch checkpoints. Learning-rate selection uses three model instances; evaluation uses ten separate instances under three force fields.

## Code, checkpoints, and experimental data

This checkout contains the reproduction source and three native TikZ diagrams. The full reproduction package, `cfc-online-adaptation-v4-code-data.zip`, additionally contains 65 checkpoints, per-instance outcomes, protocol settings, dataset manifests, validation reports, numerical figures, tables, and the dependency lockfile. It is attached as a release asset and will be archived on Zenodo with a version-specific DOI. The draft release has not been published and the DOI is not yet assigned.

For exact reproduction, download and extract the **complete package** into a new directory and follow its `README.md`. The portable reproduction entry point needs the package's checkpoints, results, and manifests; this code-only checkout is not a substitute for those data.

The public package and anonymous journal supplement contain the same experimental arrays and reproduction code. Their attribution and distribution notes differ. The bioRxiv revision will link to the GitHub release and Zenodo archive; the TMLR submission supplies an anonymous ZIP directly.

## Reproduction scope

The complete package has separate commands for checksum/model verification, numerical figure and table regeneration, all 783 online protocols, 60 compensation diagnostic conditions, fixed-readout snapshots, and offline model retraining. The additional direction-level analysis summarizes the saved results without rerunning experiments.

The recorded validation reproduces 783 online protocols and 60 diagnostic conditions, five offline models for seed 110, and six snapshot conditions. Validation reports distinguish these executed checks from the remaining available commands. All 65 checkpoint hashes and 25 numerical-source hashes were checked. The current package contains 690 checksummed files.

Use Python 3.11 and the supplied `code/uv.lock`. Keep the virtual environment outside the checkout or extracted package and outside NAS storage. Reproduction runs on CPU.

## Licenses and citation

Code is MIT-licensed; see `LICENSE` and `code/LICENSE`. The preprint is distributed under CC BY 4.0. The reserved Zenodo DOI will be inserted into the public documentation before publication. The manuscript release label `biorxiv-v4` identifies this experiment snapshot; it is independent of the Python project's internal package version.
