This is my final project for MATH 598, A topics class in Statistics which was focused on missing data. I consider the case of a dataset of images with some missing labels, I then compare the following approaches: A convolutional autoencoder and predictive mean matching at the latent level; A convolutional network which predicts class probabilities and performs predictive mean matching using the predicted probabilities; and a convolutional network which simply attempts to predict the class labels. I compare the three approaches under both the Missing Conditionally at Random (MAR) and the Missing not at Random (MNAR) cases.


This is just going to give a general overview of the different files and what they do

1) 'report.pdf'

   A PDF containing relevant work, methodology, results, and a discussion of the results

3) 'PMM_for_images.py'

    This trains the convolutional classifier and convolutional autoencoder on the CIFAR-10 data under the MAR assumption saving the models and datasets to be saved later for analysis.


4) 'w_increasing_missingness.py' 

    This performs the experiment that slowly increases the amount of missingness in the dataset and trains a convolutional autoencoder and convolutional classifier across all the different missingness settings under the MAR assumption, it then saves the datasets and models to be analysed for results later


5) 'w_increasing_missingness_mnar.py'

    This performs the experiment that checks how the convolutional autoencoder and convolutional classifier perform across different missingness-label probability distributions in the MNAR setting, by changing the alphas variable we can change the beta hyper-parameters that govern which missingness-label probabilitiy distributions get tested on. For each dataset it trains the models and then saves the models and the datasets to be analysed for results later

6) 'MNAR_case.py'

    This trains a single convolutional autoencoder and convolutional classifier on the CIFAR-10 data when the data is MNAR. It then saves the models and datasets so that analysis can be performed.

7) 'run_evals_for_paper.ipynb' 

    this is a jupyter notebook that loads every model and dataset and calculates their precision as described in the report. For each precision the first entry is the autoencoder, second is the classifier with PMM, and the third is the classifier using it's predicted probabilities.
