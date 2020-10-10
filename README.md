# Fin-Env-Narrative
This repository contains python code that was used for the experiments in the paper "A Computational Analysis of Financial and Environmental Narratives within Financial Reports and its Value for Investors"

**Note**: This repository does not contain the underlying label data and documents. The pre-processed reports can be downloaded from [SRAF](https://sraf.nd.edu/data/stage-one-10-x-parse-data/). The unprocessed files can be downloaded via the [U.S. Securities and Exchange Commission](https://www.sec.gov/Archives/edgar/Feed/). The labels (financial and environmental data) can be taken from Bloomberg. Alternatively, the environmental data can also be taken directly from Sustainalytics.

---


# Requirements
## Installation of required packages

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. Create a virtual environment and install [TensorFlow](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and [PyTorch](https://pytorch.org/get-started/locally/#start-locally).
3. Install [transformers](https://github.com/huggingface/transformers)
4. Install [pandas](https://pandas.pydata.org/docs/getting_started/install.html), [numpy](https://numpy.org/install/), [nltk](https://www.nltk.org/install.html), and [scikit-learn](https://scikit-learn.org/stable/install.html)

This repository was tested on the following versions: numpy 1.17.0, pandas 1.0.3, tensorflow 2.0.0, keras 2.3.1, nltk 3.4.5, pytorch 1.0.1, and scikit-learn 0.21.2.
Please refer to the according installation pages for the specific install command.
