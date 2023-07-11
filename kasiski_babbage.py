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


def get_index_sequences(three_or_four, cipher_text):
  '''Input a cipher text and an option for whether to look for three or four letter sequences (three_or_four)

     Returns:
     - a set of the unique distances (this is to turn the data into a pandas df - to use as the columns of the pandas dataframe)
     - A dictionary with keys the three or four letter sequences and item the spaces between their occurences as a list.'''

  cipher_text = cipher_text.replace(" ", "")

  # Get list of all three or four letter letter sequences
  if three_or_four == 3:
    letter_list = [cipher_text[i:i+3] for i in range(0,len(cipher_text)-2)]
  elif three_or_four == 4:
    letter_list = [cipher_text[i:i+4] for i in range(0,len(cipher_text)-3)]

  # collect the distances for all sequences
  unique_distances = set()

  # For each letter seq find the indices where these sequences start, put them in a list
  seq_dict = dict()

  seqs_considered = set()

  for seq in [seq for seq in letter_list if letter_list.count(seq) > 1 and seq not in seqs_considered]:
    seqs_considered.add(seq)

    seq_count = letter_list.count(seq) # get the number of occurences of the seq.
    index_list = []
    temp_index = 0
    for i in range(0, seq_count):
      pos = cipher_text[temp_index:].find(seq)
      if i == 0:
        index_list.append(pos)
        temp_index = pos+1
      else:
        index_list.append(temp_index + pos)
        temp_index = temp_index + pos + 1

    # once all starting indices of the seq has been found, compute the difference in the distances.
    distances = get_distance_between_sequences(index_list)
    # Add to the set of unique distances
    unique_distances = unique_distances.union(set(distances))

    # add list of distances to the seq dictionary
    seq_dict[seq] = distances

  return unique_distances, seq_dict

def get_distance_between_sequences(index_list):
  '''Returns the distances between the three or four letter sequences recorded in index list.
  This is a sub-routine of kasiski_babagge which takes as input, the output of get_index_sequences.'''
  upper = len(index_list)

  return [index_list[j]-index_list[i] for i in range(0,upper) for j in range(0,upper) if j > i]


def kb_dataframe(unique_distances, letter_dict):
  '''Turns the kasiski babbage data into a pandas dataframe.

     Input: - A list of unique distances between the seqs
            - A dictionary of seqs as keys and a list of distances between their repeats.
            (These are the outputs of get_index_sequences)

    Output: Pandas dataframe of the data. This is a subroutine of kasiski_babbage_df method below.
  '''
  total = []

  # create a list of lists to turn into a dataframe
  for seq in letter_dict.keys():
    seq_data = []
    seq_data.append(seq)
    for d in unique_distances:
      if d in letter_dict[seq]:
        seq_data.append(letter_dict[seq].count(d))
      else:
        seq_data.append(0)

    total.append(seq_data)

  df = pd.DataFrame(total, columns=['Sequence'] + list(unique_distances))

  return df


def kasiski_babbage_df(cipher_text, three_or_four):
  '''Returns the three AND four letter repeated sequences and the number of spaces between them.

    Output: A pandas dataframe with sequences in the first column and distances in other columns
            An entry in the table is the number of times a seq appeared at that distance apart.
  '''
  if three_or_four == 3:
    # Get dictionary of three letter sequences
    three_letter_distances, three_letter_dict = get_index_sequences(3, cipher_text)

    # combine the unique distances together
    unique_distances = list(three_letter_distances)
    unique_distances.sort()

    # turn into pandas dataframe, with unique distances as colums
    df = kb_dataframe(unique_distances, three_letter_dict)
    df.set_index('Sequence')

    return df
  
  if three_or_four == 4:
    # Get dictionary of three letter sequences
    four_letter_distances, four_letter_dict = get_index_sequences(4, cipher_text)

    # combine the unique distances together
    unique_distances = list(four_letter_distances)
    unique_distances.sort()

    # turn into pandas dataframe, with unique distances as colums
    df = kb_dataframe(unique_distances, four_letter_dict)
    df.set_index('Sequence')

    return df

  if three_or_four == 'both':
    # Get dictionary of three letter sequences
    three_letter_distances, three_letter_dict = get_index_sequences(3, cipher_text)
    four_letter_distances, four_letter_dict = get_index_sequences(4, cipher_text)

    # add the four letter info to three letter dict
    three_letter_dict.update(four_letter_dict)

    # combine the unique distances together
    unique_distances = list(three_letter_distances.union(four_letter_distances))
    unique_distances.sort()

    # turn into pandas dataframe, with unique distances as colums
    df = kb_dataframe(unique_distances, three_letter_dict)
    df.set_index('Sequence')

    return df

def kasiski_babbage_summary(df):
  '''Returns the sum of columns of kasiski babbage df as a dataframe, ordered.'''

  # Select only the numeric columns and sum their values
  numeric_cols = df.select_dtypes(include=['int64', 'float64'])
  numeric_col_sums = numeric_cols.sum().sort_values(ascending = False)
  numeric_col_sums.to_frame(name='Occurences')

  return numeric_col_sums


def repeated_sequences_yes_no(encrypted_text, min_distance=3, max_distance=6):
  """
  Determines whether a cipher text has repeated sequences. Returns 1 if there are repeated sequences and 0 if not
  """
  # Step 1: Find repeated sequences (this gives a dictionary of sequences and their starting indices as the values)
  repeated_sequences = defaultdict(list)
  for distance in range(min_distance, max_distance+1):
    for i in range(len(encrypted_text) - distance + 1):
      sequence = encrypted_text[i:i+distance]
      repeated_sequences[sequence].append(i) # appends the starting indices of appearances of the sequence.

  # Step 2: Calculate distances between repeated sequences
  for sequence, positions in repeated_sequences.items():
    if len(positions) >= 2:
      return 1

  return 0

