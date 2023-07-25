This is just going to give a general overview of the different files and what they do


1) 'PMM_for_images.py'

    This trains the convolutional classifier and convolutional autoencoder on the CIFAR-10 data under the MAR assumption saving the models and datasets to be saved later for analysis.


2) 'w_increasing_missingness.py' 

    This performs the experiment that slowly increases the amount of missingness in the dataset and trains a convolutional autoencoder and convolutional classifier across all the different missingness settings under the MAR assumption, it then saves the datasets and models to be analysed for results later


3) 'w_increasing_missingness_mnar.py'

    This performs the experiment that checks how the convolutional autoencoder and convolutional classifier perform across different missingness-label probability distributions in the MNAR setting, by changing the alphas variable we can change the beta hyper-parameters that govern which missingness-label probabilitiy distributions get tested on. For each dataset it trains the models and then saves the models and the datasets to be analysed for results later

4) 'MNAR_case.py'

    This trains a single convolutional autoencoder and convolutional classifier on the CIFAR-10 data when the data is MNAR. It then saves the models and datasets so that analysis can be performed.

5) 'run_evals_for_paper.ipynb' 

    this is a jupyter notebook that loads every model and dataset and calculates their precision as described in the report. For each precision the first entry is the autoencoder, second is the classifier with PMM, and the third is the classifier using it's predicted probabilities.