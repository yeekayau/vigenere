import time
import os
import re
import random                         # generate random strings
import string                         # do stuff to strings
import numpy as np                    # Putting stuff into rectangles (matrices)
from itertools import permutations    # To get rearrangements of stuff
from collections import defaultdict
from collections import Counter
import math
from itertools import chain
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plain_text_prep(text):
  """
  This function takes a string as input and removes all punctuation marks and spaces from it.

  Args:
  - text (str): The input string to remove punctuation from.

  Returns:
  - str: The input string without any punctuation marks.
  """
  text = text.upper()

  # Only include letters
  text_no_punct = ''.join(char for char in text if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
  text_no_punct.replace('“', "") # remove double quotation marks

  return text_no_punct.upper().replace(" ", "")

def vigenere_encrypt(plain_text, key):
  '''Encrypts a text using the vigenere cipher.

  Assumes the plain text only has letters and spaces and no punctuation.
  '''
  key = key.upper()
  key = key.replace(" ", "")
  plain_text = plain_text_prep(plain_text)
  letters_list = list(string.ascii_uppercase)

  # 26x26 grid of the alphabet (shifted by 1 place each time)
  tabular_recta = [letters_list[i:] + letters_list[0:i] for i in range(0,26)]

  # pair up the plain_text with enough copies of the key, letter by letter
  n = len(plain_text)/len(key)
  pairs = [*zip(plain_text, key*int(np.ceil(n)))]
  cipher_text = ''
  for i in pairs:
    cipher_text = cipher_text + tabular_recta[letters_list.index(i[0])][letters_list.index(i[1])]

  return cipher_text


def index_of_coincidence(text):
    '''Probability of choosing n letters from a piece of text, and the n letters are the same letter.
       When n=2, this is the index of coincidence of the text. Assumes text has been stripped of punctuation.
       '''
    # Count the number of each letter in the string
    counts = np.zeros(26)
    for char in text:
        if char.isalpha():
            index = ord(char.upper()) - ord('A')
            counts[index] += 1

    # Calculate the numerator
    numerator = np.sum(counts * (counts - 1))

    # Calculate the denominator
    l = len(text)
    denominator = l * (l - 1)

    # Check if the denominator is zero
    if denominator == 0:
        ioc = 0  # or any other value you want to assign
    else:
        # Calculate the ratio
        ioc = numerator / denominator

    return ioc

def ioc_key_length(text_length, ioc):
  '''Computes the guess for key length from IOC.'''
  return (0.028*text_length)/(ioc*(text_length -1) - 0.038*text_length + 0.066)

