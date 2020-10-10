# Data setup

To run the models, please note that you create a *dataframe* containing the name of each file, the according labels (env_q, env_t, EPS_q, BHAR_t), the market capitalization (Log_MarketCap) and industry identifier (osha_gov_SIC_numeric). So the data is in the following format:

| fname                                                         | env_q         | ...   |
| ------------------------------------------------------------- |:-------------:| -----:|
| 20150224_10-K_edgar_data_000000_0000000000-15-000000.txt      | 1             | 0     |
| 20150524_10-Q_edgar_data_000000_0000000000-15-000000.txt      | 0             | 1     |
| ...                                                           | 0             | ...   |

Further, create a folder containing all the text files you'd like to process. For running models with glove-embeddings, you also need to download the [glove-embeddings](https://nlp.stanford.edu/projects/glove/). 

## Running the code

The code of each model (BOW, CNN, etc.) is structured as follows:

1. First, the required packages are loaded.
2. Then, you need to specify the direcotry containing the *dataframe*, the text files, and the directory, where the results should be saved.
3. Each code produces its own train, test, validation split and saves them to your storage-directory. This way you can ensure that you either create your own train-test-val-split or that you always create and reload the same, unique train-test-val-split.
4. Next, the models for each label are created and stored after every epoch. To make sure each label is saved in its own folder, create sub-folders in your storage-directory named exactly like the labels (env_q, env_t, EPS_q, BHAR_t).
5. Each code contains the *direct path*, path *env*, and the *combined path*. Make sure that you assigne the according label (marked in the code with ```## << HERE:```)
