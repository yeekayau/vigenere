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

import plotly.graph_objects as go

def sorted_frequency_vector(text):
  '''returns the relative frequencies of each of the english letters in a piece of text.'''
  text_upper = text.strip().upper()
  n = len(text_upper)
  alphabet = string.ascii_uppercase

  frequency_vector = [0]*26
  if n > 0:
    frequency_vector = [text_upper.count(i)/n for i in alphabet]
    frequency_vector.sort()

  #print(frequency_vector)
  return frequency_vector

def plot_sorted_frequency_vector(vector):
  '''Plots the sorted frequency vector and compares it with the English language signature.
     
     Input: Is the sorted frequency vector
  
  '''
  english_signature = [0.0011, 0.0021, 0.0021, 0.0032, 0.0063, 0.0095, 0.0116, 0.0137, 0.0189, 0.02, 0.0211, 0.0232,
                       0.0263, 0.0316, 0.0326, 0.0337, 0.0368, 0.0421, 0.0547, 0.0663, 0.0684, 0.0747, 0.0842,
                       0.0895, 0.0979, 0.1284]
  fig = go.Figure()

  fig.add_trace(go.Scatter(x=list(range(1, len(vector) + 1)), y=vector, mode='lines', name='Input text signature',
                         line=dict(color='blue')))
  fig.add_trace(go.Scatter(x=list(range(1, len(english_signature) + 1)), y=english_signature, mode='lines', name='Language signature',
                         line=dict(color='red')))

  fig.update_layout(
    title='Sorted Relative Frequencies: English Language signature vs. Input text signature',
    xaxis_title='Index',
    yaxis_title='Relative frequency'
  )

  fig.show()

def twist(vector):
  '''returns the twist of a vector (of length 26).'''

  fourteen_twentysix = vector[13:]
  one_thirteen = vector[0:13]

  return sum(fourteen_twentysix) - sum(one_thirteen)

def split_into_cosets(text, m):
  '''Splits text into m cosets.'''
  # create a list of m empty strings.
  cosets = [''] * m

  #iterates over each character in the text and assigns each character to the appropriate coset by
  # using the modulo operator (i % m) to determine the index of the current coset.
  for i, char in enumerate(text):
    cosets[i % m] += char

  return cosets

def plot_coset_signatures(cipher_text, m):
  '''Given an integer m, plots the coset signatures.'''
  cosets = split_into_cosets(cipher_text, m)
  sorted_freq_vectors = [sorted_frequency_vector(k) for k in cosets]

  # Define the dimensions of the grid
  num_cols = 3  # Number of columns in the grid
  num_rows = m - num_cols  # Number of rows in the grid

  # Create subplots with the specified dimensions
  fig = make_subplots(rows=num_rows, cols=num_cols)

  # Iterate over the sublists and create line charts
  for i, sublist in enumerate(sorted_freq_vectors):
    row = i // num_cols + 1  # Calculate the row index
    col = i % num_cols + 1   # Calculate the column index

    # Add a line chart for the current sublist to the corresponding subplot
    fig.add_trace(go.Scatter(x=list(range(1, 27)), y=sublist, mode='lines'), row=row, col=col)

  # Update the layout and display the figure
  fig.update_layout(height=600, width=800, title_text="Line Charts Grid")
  fig.show()


def twist_of_cosets(cipher_text, m):
  '''For a given cipher text and a guess at the key length m, compute the twist of each of the m-cosets'''
  cosets = split_into_cosets(cipher_text, m)
  twists = [twist(sorted_frequency_vector(k)) for k in cosets]

  return twists

def twist_index(cipher_text, m):
  '''Combines the twist of the cosets for m into one number, per Barr and Simoson'''

  return (100/m)*sum(twist_of_cosets(cipher_text, m))

def plot_twist_index(cipher_text, guesses):
  '''Given a list of guesses of the key length, compute the twist index for each guess and plot them'''

  twists = [twist_index(cipher_text, m) for m in guesses]
  fig = go.Figure(data=go.Scatter(x=guesses, y=twists, mode='lines'))

  fig.update_layout(
    title='Twist Indices',
    xaxis_title='Guess at the key length',
    yaxis_title='Twist Index'
  )

  return fig


def twist_plus(vector):
  '''Input is a vector of twist index numbers i (say from 1 to 10)

    Output: Output is the corresponding twist_plus numbers (say from 2 to 10). The output vector has length 1 less than the input vector.
  '''

  k = len(vector)
  twist_plus_vector = [vector[i]-( (1/i)*sum(vector[j] for j in range(0, i)) ) for i in range(1, k) ]
  return twist_plus_vector

def twist_doubleplus(vector):
  '''Input is a vector of twist index numbers.
     Output: Output is the corresponding twist double plus numbers. The output vector has length 2 less than the input vector.
  '''
  k = len(vector)
  twist_doubleplus_vector = [vector[i]-0.5*(vector[i-1] + vector[i+1]) for i in range(1, k-1) ]
  return_doubleplus_vector

def max_twist_doubleplus(tdp_vector):
  '''Returns the index of the maximal local jump in twist indices. Input is the twist_doubleplus_vector.
    This is the predicted key length by twist double plus.
  '''
  np_tdp_vector = np.array(tdp_vector)
  return np.argmax(np_tdp_vector) + 2

def max_twist_plus(tp_vector):
  '''Returns the index of the maximal jump in twist indices. Input is the twist_plus_vector.
    This is the predicted key length by the twist plus.
  '''
  np_tp_vector = np.array(tp_vector)
  return np.argmax(np_tp_vector) + 2


def plot_twist_and_twistplus_twistdouble(cipher_text, guesses, twist_plus_prediction, twist_doubleplus_prediction):
    '''Given a list of guesses of the key length, compute the twist index for each guess and plot them'''

    twists = [twist_index(cipher_text, m) for m in guesses]
    fig = go.Figure(data=go.Scatter(x=guesses, y=twists, mode='lines'))

    fig.add_shape(
        type='line',
        x0=twist_plus_prediction,
        y0=0,
        x1=twist_plus_prediction,
        y1=100,
        line=dict(color='red', width=2, dash='dash'),

        x2=twist_doubleplus_prediction,
        y2=0,
        x3=twist_doubleplus_prediction,
        y3=100,
        line=dict(color='green', width=2, dash='dash')
        
    )

    fig.update_layout(
        title='Twist Indices',
        xaxis_title='Guess at the key length',
        yaxis_title='Twist Index'
    )

    fig.update_layout(
        title='Twist Indices',
        xaxis_title='Guess at the key length',
        yaxis_title='Twist Index',
        annotations=[
            go.layout.Annotation(
                x=twist_plus_prediction,
                y=100,
                xref="x",
                yref="y",
                text="Twist+ Prediction",
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            ),
            go.layout.Annotation(
                x=twist_doubleplus_prediction,
                y=100,
                xref="x",
                yref="y",
                text="Twist++ Prediction",
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            )
        ]
    )

    return fig

