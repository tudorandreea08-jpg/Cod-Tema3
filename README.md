# Cod-Tema3
Codul sursă utilizat pentru preprocesarea datelor și extragerea regulilor de asociere.
# Association Rule Mining on FAO/WHO GIFT Data

This repository contains the source code used for preprocessing food consumption data
from the FAO/WHO GIFT platform and for extracting association rules using the FP-Growth algorithm.

## Dataset
The code was tested on microdata from the FAO/WHO Global Individual Food Consumption Data Tool (GIFT),
using the Romanian national dietary survey.

Due to data usage restrictions, the dataset is not included in this repository.
The file `consumption_user.csv` must be downloaded separately from:
https://www.fao.org/gift-individual-food-consumption/data/en

## Methodology
- Transaction definition: foods consumed by one individual in one recall day
- Algorithm: FP-Growth
- Parameters:
  - min_support = 0.01
  - min_confidence = 0.5
  - lift > 1

## How to run

1. Install dependencies:
```bash
pip install -r requirements.txt
