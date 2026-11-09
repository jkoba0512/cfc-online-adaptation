# Readout adaptation in simulated arm reaching

Complete code and data for **Readout adaptation in simulated arm reaching: learned compensation and performance after updates stop**, by Jun Kobayashi.

This manuscript release compares six adaptive variants: dense-backbone CfC LMS, NCP-wired CfC LMS, FC-wired CfC LMS, ESN LMS, RBF LMS, and dense-CfC RLS. All three CfC configurations use final 400-epoch checkpoints. Learning-rate selection uses three model instances; evaluation uses ten separate instances under three force fields.

## Reproduce the results

This repository contains the complete six-method reproduction dataset: source code, 65 checkpoints, individual results, manifests, dependency lock, validation reports, figures, tables, and three native TikZ diagrams. Follow **[REPRODUCTION.md](REPRODUCTION.md)** for environment setup, executed validation scope, and reproduction commands. Keep the Python environment outside the checkout and NAS storage. Use Python 3.11.

Start by verifying the complete checkout:

```bash
python scripts/reproduce_rerun400_package.py --stage verify --out checks
```

Use the Python environment described in REPRODUCTION.md. The portable entry point also supports figure regeneration, 783 online protocols, 60 diagnostic reruns, fixed-readout snapshots, and offline model retraining. The direction-level analysis reaggregates saved results without running new experiments.

## Archive and publication status

The release branch contains the checkpoints and data themselves, not only external release attachments. After the repository is public and enabled in Zenodo, publishing the GitHub release will trigger archival of this complete repository snapshot and assignment of a version-specific DOI. DOI reservation or a separate manual Zenodo upload is not required. For a published release, use the version-specific Zenodo DOI recorded in the GitHub release notes.

The attached `cfc-online-adaptation-v4-code-data.zip` is an additional direct-download package containing the same scientific data and reproduction code. Its README and package checksums differ from the repository's documentation and checksum inventory. GitHub's source archive, which Zenodo archives, is also self-contained.

The anonymous journal supplement contains the same experimental arrays and reproduction code. Its attribution and distribution descriptions differ. The bioRxiv revision will cite the GitHub release and Zenodo archive, while TMLR receives an anonymous ZIP directly.

## Licenses and citation

Code is MIT-licensed; see LICENSE and code/LICENSE. The preprint is distributed under CC BY 4.0. Zenodo metadata is supplied in `.zenodo.json`; `CITATION.cff` identifies the software author and release. The label `biorxiv-v4` identifies this experiment snapshot independently of the Python package's internal version.
