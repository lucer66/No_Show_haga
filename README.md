# No_Show

Authors: Ruben Peters, Ingmar Loohuis
Email: r.peters-7@umcutrecht.nl

this project by the UMC was adapted to fit the context of the HagaZiekenhuis by Luc van de Wal

## Installation

To install the noshow package use:

```{bash}
pip install -e .
```

## Run pipelines

The no-show code uses DVC pipelines. To run the feature-building and model stages, use:

```{bash}
dvc pull
dvc repro
```
