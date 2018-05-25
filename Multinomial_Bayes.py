'__author__ Rumela'
from __future__ import print_function
import os, math, sys, codecs

class BayesText:
    def __init__(self, training_dir, stopwords_file):
        """
        This is the initialization class which takes 3 parameters:
        training_path : path of training data. The training path contains all the
                        directories that have the training data. Each directory is the
                        class of all the words contained in its files
        stopwords : List of stopwords. Stop words essentially improves the accuracy
                    so I included stop words after consulting with professor. Stop words
                    occur like 1word/line. I got this list from the internet
        ignore_words : List of words that can safely be ignored
        """
        self.vocabulary = {}
        self.probability = {}
        self.sum = {}
        self.stopwords = {}
        stopword_file = open(stopwords_file)
        for line in stopword_file:
            self.stopwords[line.strip()] = 1
        stopword_file.close() # close the stop words file

        self.categories = os.listdir(training_dir)#list of all directories in training directory
        # remove anything from the directory which is not a directory
        #self.categories = [filename for filename in categories if os.path.isdir(training_dir + filename)]

        print("The categories for which we are finding the class are --> ")
        for category in self.categories:
            print('    ' + category)
            (self.probability[category], self.sum[category]) = self.train(training_dir, category)
        #If any word does nto occur more than 5 times it will be eliminated here
        delete_list = []
        for word in self.vocabulary:
            if self.vocabulary[word] < 5:
                delete_list.append(word)
        for word in delete_list:
            del self.vocabulary[word]
        # Computing the probabilties for the vocabulary
        len_vocab = len(self.vocabulary)
        print("Computing the probabilities --> ")
        for category in self.categories:
            print('    ' + category)
            factor = self.sum[category] + len_vocab
            for word in self.vocabulary:
                if word in self.probability[category]:
                    count = self.probability[category][word]
                else:
                    count = 1
                self.probability[category][word] = (float(count + 1) / factor)
        print("Training phase completed. Moving on to testing --> \n\n")

    #Find the category of a word by taking the log of the probability as described in text
    def NB_Classification(self, filename):
        result = {}
        for category in self.categories:
            result[category] = 0
        f = codecs.open(filename, 'r', encoding="iso8859-1")
        for line in f:
            tokens = line.split()
            for token in tokens:
                token = token.strip('\'".,?:-').lower()
                if token in self.vocabulary:
                    for category in self.categories:
                        if self.probability[category][token] == 0:
                            print("%s %s" % (category, token))
                        result[category] += math.log(self.probability[category][token])
        f.close()
        result = list(result.items())
        result.sort(key=lambda tuple: tuple[1], reverse=True)
        return result[0][0]

    """This method is called to count the word occurences for a particular category"""

    def train(self, training_dir, category):
        currentdir = os.path.join(training_dir, category)
        files = os.listdir(currentdir)
        counts = {}
        total = 0
        for file in files:
            f = codecs.open(currentdir + '/' + file, 'r', 'iso8859-1')
            for line in f:
                tokens = line.split()
                for token in tokens:
                    token = token.strip('\'".,?:-')  # Stripping puntuation
                    token = token.lower()  # Stripping lowercase tokens
                    if token != '' and not token in self.stopwords:
                        self.vocabulary.setdefault(token, 0)
                        self.vocabulary[token] += 1
                        counts.setdefault(token, 0)
                        counts[token] += 1
                        total += 1
            f.close()
        return (counts, total)

    def testing(self, directory, category):
        files = os.listdir(directory)
        total = 0
        correct = 0
        for file in files:
            total += 1
            files_in_dir = os.path.join(directory, file)
            result = self.NB_Classification(files_in_dir)
            if result == category:
                correct += 1
        return (correct, total)
    """This method tests the files in the subdirectories."""
    def test(self, testdir):
        self.categories = os.listdir(testdir)
        correct = 0
        total = 0
        for category in self.categories:
            print(".", end="")
            cur_dir = os.path.join(testdir, category)
            (catCorrect, catTotal) = self.testing(cur_dir, category)
            correct += catCorrect
            #print (correct)
            total += catTotal
            #print (total)
        try:
            print("\n\nAccuracy is  %f%%  (%i Test Instances)" % ((float(correct) / total) * 100, total))
        except ZeroDivisionError:
            print("Division by zero!")

#Enter all path as input
base_dir = sys.argv[1]
training_dir = sys.argv[2]
test_dir = sys.argv[3]
stopwords_file = sys.argv[4]
bT = BayesText(training_dir, stopwords_file)
print("Testing Naive Bayes for Text Classification --> ")
bT.test(test_dir)
