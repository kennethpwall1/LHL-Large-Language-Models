import os
import pandas as pd
import pickle
import numpy as np
import string
import nltk
import re
from nltk.corpus import stopwords
from transformers import pipeline
import accuracy
from datasets import load_dataset, load_metric
import evaluate

def remove_punctuation(review):
    """
    Removes all the punctuation from a string and makes it lower case.

    Parameters:
    review (string): a movie review

    Returns:
    String: a movie review string in lower-case with punctuation removed.
    """
    punctuation_removed = ''
    review = review.lower()
    for r in review:
        if r not in string.punctuation:
            punctuation_removed += r
    return punctuation_removed

def remove_stopwords_tokens(tokens):
    """
    Removes all stop words from a list of tokens

    Parameters:
    tokens (list): a movie review list of words

    Returns:
    list: a movie review list of tokens with stop words removed.
    """
    stop_words = stopwords.words('english')
    stop = [word for word in tokens if word not in stop_words]
    return stop

def remove_stopwords_text(text):
    """
	Removes all stop words from a string of text

	Parameters:
	text (string): a movie review string of words

	Returns:
	string: a movie review string with stop words removed.
	"""
    stop_words = stopwords.words('english')
    stop_words_pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
    cleaned_text = stop_words_pattern.sub('', text)
    return cleaned_text

def tokenize(review):
    """
    Creates a list of word tokens from a movie review string

    Parameters:
    review (string): a movie review string of words

    Returns:
    string: a movie review string with stop words removed.
    """
    lower = review.lower()
    tokens = lower.split()
    return tokens

def get_sentiment(classifier, text):
    """
    Provides sentiment positive/negative with the probability of a movie review.

    Parameters:
    classifier (classifier): Pipeline BERT model
    text (string): a movie review string of words

    Returns:
    list: with dictionary inside including the following key value pairs:
    - label: negative/positive
    - score: 0.9999
    """
    return classifier(text)

def truncate_string(text, max_length=512):
    """
    truncates a movie review string to 512 characters required to use BERT model

    Parameters:
    text (string): a movie review string of words

    Returns:
    string: a movie review string truncated to 512 characters.
    """
    if len(text) > max_length:
        return text[:max_length]
    else:
        return text

def preprocess_function(tokenizer, examples):
    """
    Tokenizes a string of text

    Parameters:
    tokenizer (tokenizer)
    examples (string): a movie review string of words

    Returns:
    datasetDict: tokenized imdb movie reviews 
    """
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

def compute_metrics(p):
    """
    The compute_metrics function is designed to evaluate the performance of a 
    machine learning model, specifically by calculating the accuracy of the model's predictions. 

    Parameters:
    p (dictionary) - the tokenized train and test data passed to the trainer

    Returns:
    float: return metrics accuracy during training and evaluation
    
    """
    metric = evaluate.load("accuracy")
    preds = np.argmax(p.predictions, axis=1)
    accuracy = metric.compute(predictions=preds, references=p.label_ids)
    return accuracy