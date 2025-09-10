\# Modern Power Cube — Transformer-based Electricity Generation Forecasting (PoC)



\*\*Short description\*\*

A proof-of-concept implementation of the "Modern Power Cube" framework for long-term electricity generation forecasting using Transformer-based models. This repository is a browsable copy of the official release archived on Zenodo (see DOI below).



---



\## Official record / Citation

This project is officially published on Zenodo:

\*\*DOI:\*\* \https://doi.org/10.5281/zenodo.17086721  

Citation:

> Alghamdi, Abdulaziz (2025). Modern Power Cube — Transformer-based electricity forecasting (PoC). Zenodo. DOI: \[INSERT\_ZENODO\_DOI\_HERE]



---



\## What this repository contains

\- `data/` — (not included) metadata and pointers to the official dataset (25-year annual series). See \*Data Access\* below.

\- `src/` — source code for data processing, model training, evaluation, and plotting.

\- `notebooks/` — runnable Jupyter notebooks demonstrating the pipeline:

&nbsp; - `00\_data\_overview.ipynb`

&nbsp; - `01\_model\_baselines.ipynb`

&nbsp; - `02\_transformer\_experiment.ipynb`

\- `results/` — model outputs, evaluation tables, and figures (sample).

\- `LICENSE` — Apache-2.0 (same as Zenodo release).

\- `README.md` — you are reading it.



---



\## Quick start (run the PoC locally)

\### Requirements

\- Python 3.10+  

\- Create virtual env and install:

```bash

python -m venv venv

source venv/bin/activate   # or `venv\\Scripts\\activate` on Windows

pip install -r requirements.txt



