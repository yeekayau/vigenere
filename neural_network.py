import numpy as np
import pandas as pd
import tensorflow as tf

import vigenere_encrypt as ve
import twist as tw
import kasiski_babbage as kb


def average_coset_coincidences(cipher_text):
  '''
  Input: a cipher text
  Output: a vector of length 23 with entries being the average coset coincidences for i between 3 and 26
  '''
  average_coset_coincidence = []
  for i in range(3,27):
      i_cosets = tw.split_into_cosets(cipher_text, i)

      average_coset_coincidence.append(np.mean([ve.index_of_coincidence(k) for k in i_cosets if len(k) > 0]))

  return average_coset_coincidence


def top_5_distances(cipher_text, kb_df):
  '''Input: The Kasiski babbage dataframe
     Output: a list of the column labels of the top 5 most common distances overalls (includes repeats of the same seq at the same distances)
  '''

  kb_df = kb_df

  column_sums = kb_df.iloc[:, 1:].sum()  # Calculate the sum of each column
  top_5_columns = column_sums.nlargest(5)  # Get the top 5 columns with the highest sums

  # Get the labels of the top 5 columns
  top_5_column_labels = top_5_columns.index.tolist()

  if len(top_5_column_labels) < 5:
    top_5_column_labels = top_5_column_labels + [0]*(5 - len(top_5_column_labels))

  return top_5_column_labels


def how_many_times_top_5_distances_appeared(cipher_text, kb_df):
  '''Input: The Kasiski babbage dataframe
     Output: a list of the column labels of the top 5 most common distances overalls (includes repeats of the same seq at the same distances)
  '''

  kb_df = kb_df

  column_sums = kb_df.iloc[:, 1:].sum()  # Calculate the sum of each column
  top_5_columns = column_sums.nlargest(5)  # Get the top 5 columns with the highest sums
  top_5_columns = top_5_columns.tolist()

  if len(top_5_columns) < 5:
    top_5_columns = top_5_columns + [0]*(5 - len(top_5_columns))

  return top_5_columns


def get_info_for_nn(cipher_text, kb_df):
  '''Given a cipher text, returns a np.array of the features needed to plug it into the NN model.'''
  
  kb_df = kb_df
  data = []

  data.append(len(cipher_text))
  data.append(ve.index_of_coincidence(cipher_text))
  data.append(kb.repeated_sequences_yes_no(cipher_text))
  data.append(len(cipher_text) // 12)

  twist_indices = [tw.twist_index(cipher_text, i) for i in range(1,29)]
  data.extend(twist_indices)

  twist_p = tw.twist_plus(twist_indices)
  data.extend(twist_p)

  avg_coset_ioc = average_coset_coincidences(cipher_text)
  data.extend(avg_coset_ioc)

  data.extend(top_5_distances(cipher_text, kb_df))
  data.extend(how_many_times_top_5_distances_appeared(cipher_text, kb_df))

  return np.array(data)

def make_prediction_on_single_sample(cipher_text, kb_df, model):
	'''
	Make a prediction. 
	'''
	# Extract info from Cipher text
	test = get_info_for_nn(cipher_text, kb_df)
		
	# make prediction!
	test = tf.convert_to_tensor(test, dtype=tf.float32)
	test = tf.reshape(test, (1, -1))

	prediction = model.predict(test)
	prediction = prediction.tolist() # convert the numpy array to list
	prediction = prediction[0] # the probablities are the inner list

	top_n = 3

	# Sort the list in descending order and get the indices of the top N values
	top_indices = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)[:top_n]

	top_indices_probabilities = [prediction[i] for i in top_indices]

	top_indices = np.array(top_indices) + 3

	return top_indices, top_indices_probabilities

