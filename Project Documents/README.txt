This project has seevral parts.

1) Multinomial Naive Bayes execution steps
	i) Open the Multinomial_Bayes.py file using the IDE.
	ii) Pass 4 parameters in the following order:
	   a) Base directory
	   b) Training directory - Meaning the path of the training directory 
	   c) Test directory - Meaning the path of the Test directory
	   d) Stopwords file - Path of the stop words file

2) Steps to run the CNN file.
Sample test data is included for reference.
Train the model:
run the python script : train.python

Evaluate the model on the test data:
After training the model when we write the model learning in the checkpoints we then use it to evaluate the result.
run the python script : eval.py --eval_train --checkpoint_dir="./runs/input_the_checkpoint_folder_number/checkpoints/"

3) RNN
Input Filse
===========
csv files are needed from kaggle:
https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags/data

Libraries Required
==================
python - scikit-learn, pandas, numpy, tensorflow, keras, nltk, bs4, gensim

Instrucyions to run
===================
1. python prepro.py
2. python RNN.py
3. python calculate-f1-score.py
4. python tag-stats.py

1 - for preprocessing and building word vector
	- requires all the input csv files
	- gives the wordvec model as output which is saved in the local directory
2 - this will build recurrent neural network
	- above obtained wordvec model is loaded
	- gives the rnn model which is required for the f2 score calculation
3 - calculate the f1 score
	- requires above rnn model output
4 - statistics on the tags present in all the csv files
	- need all the input csv files
