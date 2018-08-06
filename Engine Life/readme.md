# Engine Life Prediction

A comparison of automated feature engineering using Featuretools and manual feature engineering
for predicting the remaining useful life of an engine. 

![](images/featuretools_mostimportant.png)

### Notebooks 
The notebooks for this project are:

1.  `Manual Engine Life.ipynb`
2. `Automated Engine Life.ipynb`

The `utils.py` script contains a number of useful helper functions.

### Data

The data is from the NASA Turbofan Engine Degradation Simulation Data Set
and is available [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)

To run the notebooks, place the following files in the input directory:
`train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
