import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

import vigenere_encrypt as ve
import twist as tw
import kasiski_babbage as kb
import neural_network as nn

st.title("Finding the Key Length of a Vigenere Cipher")

st.sidebar.markdown("This small web application is based on the research project **_Finding the key length of a Vignere cipher_** at Furman University. The application allows you to either supply a piece of plain text along with a key, or a piece of Vigenere encrypted cipher text. The application then displays a number of statistics related to finding the key length of the cipher text.")

intro_text2 = st.sidebar.markdown("Notably, some newer statistics from recent literature is included, as well as a neural network.")

intro_text3 = st.sidebar.markdown("This is joint work with Morgan Carns, Christian Millichap, Alyssa Pate and [Yeeka Yau](https://yeekayau.github.io/).")


choice = st.radio("Pick one", ["Encrypt my plain text with a given key", "I already have Vigenere encrypted text"])

if choice == "Encrypt my plain text with a given key":

	plain_text = st.text_area("Enter plain text")
	key = st.text_input("Enter the key for encryption")

	if plain_text and key:

	######## Display Cipher text

		st.header("This is the Cipher text")

		cipher_text = ve.vigenere_encrypt(plain_text, key.upper())
		cipher_text

	####### Display Index of coincidence

		st.header("Index of Coincidence")

		st.markdown("The Index of Coincidence is the probability that two letters picked at random from a ciphertext are the same. It is defined as follows:")

		st.latex(r'IC = \frac{1}{{l \choose 2}} \sum_{i = 1}^{26} {n_i \choose 2}')

		st.latex(r'''where $n_i$ is the number of occurrences of the letter $i$ in the Ciphertext and $l$ is the length of the ciphertext.''')

		st.markdown('**The Key Idea**')

		bullet_points = [
    					'In a piece of regular English text, $IC \approx 6.5\%$ (0.065).',
    					'In the Vigenere cipher, $IC$ is lower because a single letter can be encrypted by different letters throughout the ciphertext (the goal of this cipher is so that frequency analysis cannot be used).',
    					'So the IOC generally tells you whether you have a monoalphabetic cipher or a polyalphabetic cipher.'
			]

		bullet_list = '\n'.join([f'- {item}' for item in bullet_points])

		st.markdown(bullet_list)

		st.text("The Index of Coincidence for the cipher text is: ")
		ioc = ve.index_of_coincidence(cipher_text)
		ioc

	####### Display Kasiski Babbage table

		st.header("Kasiski Babbage Test")

		kb_df = kb.kasiski_babbage_df(cipher_text, 'both')
		kb_df

	####### Display Twist and Twist+ info

		st.header("Twist Index and Twist+")
		# twist index numbers
		tw_nums = [tw.twist_index(cipher_text, i) for i in range(3,26)]
		# twist plus numbers
		twplus_nums = tw.twist_plus(tw_nums)
		# max of twist plus numbers
		twist_plus_prediction = tw.max_twist_plus(twplus_nums)

		# chart

		twist_fig = tw.plot_twist_and_twistplus(cipher_text, [i for i in range(3,26)], twist_plus_prediction)
		st.plotly_chart(twist_fig)

	####### Display the prediction by the Neural network

		st.header("Key length Prediction by Neural Network")

		# Load the model
		model = load_model("first_model.h5")

		top_3, probs = nn.make_prediction_on_single_sample(cipher_text, kb_df, model)

		st.text("Here are the top 3 predictions by the neural network and their asscociated probabilities.")
		*zip(top_3, probs)

#################################################################
elif choice == "I already have Vigenere encrypted text":

	cipher_text = st.text_input("Enter your vigenere encrypted text")

	if cipher_text:

	####### Display Index of coincidence

		st.header("Index of Coincidence")
		ioc = ve.index_of_coincidence(cipher_text)
		ioc

	####### Display Kasiski Babbage table

		st.header("Kasiski Babbage Test")

		kb_df = kb.kasiski_babbage_df(cipher_text, 'both')
		kb_df

	####### Display Twist and Twist+ info

		st.header("Twist Index and Twist+")

		# vector of twist index numbers
		tw_nums = [tw.twist_index(cipher_text, i) for i in range(3,26)]
		# Twist plus numbers
		twplus_nums = tw.twist_plus(tw_nums)
		# max of the twist plus numbers
		twist_plus_prediction = tw.max_twist_plus(twplus_nums)

		# chart

		twist_fig = tw.plot_twist_and_twistplus(cipher_text, [i for i in range(3,26)], twist_plus_prediction)
		st.plotly_chart(twist_fig)

	####### Display the prediction by the Neural network

		st.header("Key length Prediction by Neural Network")

		# Load the model
		model = load_model("al_model_extra_layer.h5")

		top_3, probs = nn.make_prediction_on_single_sample(cipher_text, kb_df, model)

		st.text("Here are the top 3 predictions by the neural network and their associated probabilities.")
		*zip(top_3, probs)

		


