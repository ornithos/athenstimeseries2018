# athenstimeseries2018
Time series excercises for the ACM data science summer school - Athens 2018

These exercises are designed to provide a review of some common AR modelling commands in python and a brief look at time series classification. All exercises are contained in the notebook (`time-series.ipynb`) but note that submissions are required in a separate .txt or .pdf file (not .doc[x]?).

We assume that Python 3 + Anaconda are installed; furthermore the following packages are required:

* numpy
* pandas
* statsmodels
* matplotlib
* scikit-learn
* jupyter

Have fun!


### Troubleshooting

1. **I don't have Anaconda/Python3**. We highly recommend installing Anaconda. See installation instructions [here](https://conda.io/docs/user-guide/install/index.html). You're welcome to use Python without Anaconda, but you're on your own if you do!

2. **I don't have the above packages / I'm not sure I have the above packages**. It's often sensible to set up a virtual environment for new package requirements. 
    * **I know what a virtual environment is, and no, I don't want a new one**. Sure, activate an existing virtual environment and install the above packages using `conda install <package name(s)>`.
    * **I don't know what a virtual environment is, or I want to create a new one**. The following command should generate an environment called `timeseries` with the required packages: 
    
 ```
 conda create -n timeseries python=3 matplotlib numpy pandas statsmodels scikit-learn jupyter
 ```

3. ** How do I run the notebook?**
Let's call the virtual environment in step 2 `<env name>`. 
Jupyter can then be launched from either:
    * **Anaconda Navigator**: please select `<env name>` from the dropdown menu at the top ("applications on..." or similar) _first_, and then launch jupyter notebook; or
    * **Command line**: by activating the environment and then launching jupyter. This can be achieved with the following two lines at the command line:
```shell
conda activate <env name>
jupyter notebook
```

 The notebook server should appear in your browser. Simply navigate to the (this) notebook file and open it.

--------------