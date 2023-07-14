import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

import vigenere_encrypt as ve
import twist as tw
import kasiski_babbage as kb
import neural_network as nn

st.title("Finding the Key Length of a Vigen\u00E8re Cipher")

st.sidebar.markdown("This small web application is based on the research project **_Finding the key length of a Vigen\u00E8re cipher_** at Furman University. The application allows you to either supply a piece of plain text along with a key, or a piece of Vigen\u00E8re encrypted cipher text. Once input is supplied, the application displays a number of statistics related to finding the key length of the cipher text.")

intro_text2 = st.sidebar.markdown("Notably, some newer statistics from recent literature is included (see the section on Twist and Twist+). We have also trained a Feed Forward Neural Network to predict the key length (see the section on the Neural Network for details.)")

intro_text3 = st.sidebar.markdown("This is joint work with [Morgan Carns](https://www.linkedin.com/in/morgan-carns-aa4b26238/), [Christian Millichap](https://sites.google.com/view/christianmillichap/home), Alyssa Pate and [Yeeka Yau](https://yeekayau.github.io/).")


choice = st.radio("Pick one", ["Encrypt my plain text with a given key", "I already have Vigen\u00E8re encrypted text"])

if choice == "Encrypt my plain text with a given key":

	plain_text = st.text_area("Enter plain text")
	key = st.text_input("Enter the key for encryption (do not include any punctuation)")

	if plain_text and key:

	######## Display Cipher text

		st.header("This is your cipher text")

		cipher_text = ve.vigenere_encrypt(plain_text, key.upper())
		cipher_text

	####### Display Index of coincidence

		st.header("Index of Coincidence")

		with st.expander("Details"):
			st.markdown("The Index of Coincidence (IC) is the probability that two letters picked at random from a ciphertext are the same. It is defined as follows:")


			st.latex(r'''
					IC = \frac{1}{ \binom{l}{2} }  \sum_{i = 1}^{26} \binom{n_i}{2}
					''')

			st.markdown("where $n_i$ is the number of occurrences of the letter $i$ in the Ciphertext and $l$ is the length of the ciphertext.")

			st.markdown('**The Key Idea**')


			bullet_points = [
    					'In a piece of regular English text, $IC \\approx 6.5\\%$ (0.065).',
    					'In Vigen\u00E8re cipher text, the $IC$ is lower because a single letter can be encrypted by different letters throughout the ciphertext (the goal of this cipher is so that frequency analysis cannot be used).',
    					'One can then compare the IC of the cipher text to statistically established values of the IC for texts containing varying numbers of alphabets.'
			]

			bullet_list = '\n'.join([f'- {item}' for item in bullet_points])

			st.markdown(bullet_list)

			st.markdown("Alternatively, one can also use a formula derived from the IC as a predictor of key length:")

			st.latex(r'''
					L = \frac{0.028N}{(IC)(N-1)-0.038N + 0.066}
				''')

		st.text("The Index of Coincidence for the cipher text is: ")
		ioc = ve.index_of_coincidence(cipher_text)
		ioc

		st.text("Thus, the predicted key length by formula is:")
		ioc_prediction = ve.ioc_key_length(len(cipher_text), ioc)
		ioc_prediction

	####### Display Kasiski Babbage table

		st.header("Kasiski Babbage Test")

		with st.expander("Details"):
			st.markdown("The key idea of the Kasiski-Babbage test is to look for repeated sequences of length 3 or more in the cipher text, with the assumption that a repeated sequence may indicate that some of the same repeated letters of the plain text line up with letters in the key in the same position.")
			st.markdown("Based on this assumption, the distance between these repeated sequences in the cipher text would then be a multiple of the key length.")
			st.markdown("This technique appears to be remarkably effective when a cipher text contains repeated sequences. (see the paper [Kasiski's test: Couldn't the Repetitions be by Accident?](https://www.tandfonline.com/doi/full/10.1080/01611190600803819) for some theoretical reasons why this assumption is plausible.)")

		st.markdown("This table displays the repeated sequences found in the Cipher text. The columns indicate the distances apart in which they appeared, and the entries indicate how many times the repeated sequence appeared at those distances.")

		kb_df = kb.kasiski_babbage_df(cipher_text, 'both')
		kb_df

	####### Display Twist and Twist+ info

		st.header("Twist Index and Twist+")

		with st.expander("Details"):

			st.markdown("Let $C = [c_1, c_2, ... , c_{26}]$ be the sorted (least to most) relative frequencies of letters in a text (called the signature of a text)").

			st.markdown("The *twist* of $C$ is:")

			st.latex(r'''
				 \diamond C = \sum_{i = 14}^{26}c_i - \sum_{i = 1}^{13}c_i 
				''')

			st.markdown('Roughly, you can visualise this as the difference in the areas of the last 13 trapezoids and the first 13 trapezoids in a sorted relative frequency vector of the 26 letters.')

			st.markdown('The larger this number is, the more closely the signature of the text matches that of regular english (and thus a monoalphabetic cipher). If this number is close to 0, then that indicates a more uniform distribution of letters, suggesting a polyalphabetic cipher.')

			st.markdown('The idea is then to compute the twist of cosets of letters for different guesses at the key length $m$. The largest of the twist metrics (called the *twist index*) should be a candidate for the key length. The Twist of Ciphertext $\mathscr{M}$ with the guess of key length $m$ is:')

			st.latex(r'''
				 T(\mathscr{M}, m) = round \big( \frac{100}{m} \sum_{i = 1}^{m} \diamond C \big) 
				''')

			st.markdown("(see the papers [Twisting the Keyword Length from a Vigenère Cipher](https://www.tandfonline.com/doi/full/10.1080/01611194.2014.988365) and [How to improve the twist algorithm](https://www.tandfonline.com/doi/full/10.1080/01611194.2019.1657202) ) for further details.")

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

		with st.expander("Details"):
			st.markdown("We trained a Feed Forward Neural Network with 2 hidden layers of 64 neurons each and 91 input features to predict the key length of a cipher text. The network was trained on approximately 500,000 samples of Cipher texts (from books downloaded from the online book repository [www.gutenberg.org](www.gutenberg.org)) between 100 and 500 characters in length, with keys ranging in length from 3 to 25. The features include various statistics related to the methods described on this webpage, and will be detailed in forthcoming work.")

		# Load the model
		model = load_model("first_model.h5")

		top_3, probs = nn.make_prediction_on_single_sample(cipher_text, kb_df, model)

		st.markdown("Here are the top 3 predictions by the neural network and their asscociated probabilities.")

		*zip(top_3, probs)

#################################################################
elif choice == "I already have Vigen\u00E8re encrypted text":

	cipher_text = st.text_input("Enter your vigenere encrypted text")

	if cipher_text:

	####### Display Index of coincidence

		st.header("Index of Coincidence")

		with st.expander("Details"):
			st.markdown("The Index of Coincidence (IC) is the probability that two letters picked at random from a ciphertext are the same. It is defined as follows:")


			st.latex(r'''
					IC = \frac{1}{ \binom{l}{2} }  \sum_{i = 1}^{26} \binom{n_i}{2}
					''')

			st.markdown("where $n_i$ is the number of occurrences of the letter $i$ in the Ciphertext and $l$ is the length of the ciphertext.")

			st.markdown('**The Key Idea**')


			bullet_points = [
    					'In a piece of regular English text, $IC \\approx 6.5\\%$ (0.065).',
    					'In Vigen\u00E8re cipher text, the $IC$ is lower because a single letter can be encrypted by different letters throughout the ciphertext (the goal of this cipher is so that frequency analysis cannot be used).',
    					'One can then compare the IC of the cipher text to statistically established values of the IC for texts containing varying numbers of alphabets.'
			]

			bullet_list = '\n'.join([f'- {item}' for item in bullet_points])

			st.markdown(bullet_list)

			st.markdown("Alternatively, one can also use a formula derived from the IC as a predictor of key length:")

			st.latex(r'''
					L = \frac{0.028N}{(IC)(N-1)-0.038N + 0.066}
				''')

		st.text("The Index of Coincidence for the cipher text is: ")
		ioc = ve.index_of_coincidence(cipher_text)
		ioc

		st.text("Thus, the predicted key length by formula is:")
		ioc_prediction = ve.ioc_key_length(len(cipher_text), ioc)
		ioc_prediction

	####### Display Kasiski Babbage table

		st.header("Kasiski Babbage Test")

		with st.expander("Details"):
			st.markdown("The key idea of the Kasiski-Babbage test is to look for repeated sequences of length 3 or more in the cipher text, with the assumption that a repeated sequence may indicate that some of the same repeated letters of the plain text line up with letters in the key in the same position.")
			st.markdown("Based on this assumption, the distance between these repeated sequences in the cipher text would then be a multiple of the key length.")
			st.markdown("This technique appears to be remarkably effective when a cipher text contains repeated sequences. (see the paper [Kasiski's test: Couldn't the Repetitions be by Accident?](https://www.tandfonline.com/doi/full/10.1080/01611190600803819) for some theoretical reasons why this assumption is plausible.)")

		st.markdown("This table displays the repeated sequences found in the Cipher text. The columns indicate the distances apart in which they appeared, and the entries indicate how many times the repeated sequence appeared at those distances.")

		kb_df = kb.kasiski_babbage_df(cipher_text, 'both')
		kb_df

	####### Display Twist and Twist+ info

		st.header("Twist Index and Twist+")

		with st.expander("Details"):

			st.markdown("Let $C = [c_1, c_2, \ldots , c_{26}]$ be the sorted (least to most) relative frequencies of letters in a text (called the signature of a text)").

			st.markdown("The *twist* of $C$ is:")

			st.latex(r'''
				 \diamond C = \sum_{i = 14}^{26}c_i - \sum_{i = 1}^{13}c_i 
				''')

			st.markdown('Roughly, you can visualise this as the difference in the areas of the last 13 trapezoids and the first 13 trapezoids in a sorted relative frequency vector of the 26 letters.')

			st.markdown('The larger this number is, the more closely the signature of the text matches that of regular english (and thus a monoalphabetic cipher). If this number is close to 0, then that indicates a more uniform distribution of letters, suggesting a polyalphabetic cipher.')

			st.markdown('The idea is then to compute the twist of cosets of letters for different guesses at the key length $m$. The largest of the twist metrics (called the *twist index*) should be a candidate for the key length. The Twist of Ciphertext $\mathscr{M}$ with the guess of key length $m$ is:')

			st.latex(r'''
				 T(\mathscr{M}, m) = round \big( \frac{100}{m} \sum_{i = 1}^{m} \diamond C \big) 
				''')

			st.markdown("(see the papers [Twisting the Keyword Length from a Vigenère Cipher](https://www.tandfonline.com/doi/full/10.1080/01611194.2014.988365) and [How to improve the twist algorithm](https://www.tandfonline.com/doi/full/10.1080/01611194.2019.1657202) ) for further details.")

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

		with st.expander("Details"):
			st.markdown("We trained a Feed Forward Neural Network with 2 hidden layers of 64 neurons each and 91 input features to predict the key length of a cipher text. The network was trained on approximately 500,000 samples of Cipher texts (from books downloaded from the online book repository [www.gutenberg.org](www.gutenberg.org)) between 100 and 500 characters in length, with keys ranging in length from 3 to 25. The features include various statistics related to the methods described on this webpage, and will be detailed in forthcoming work.")

		# Load the model
		model = load_model("al_model_extra_layer.h5")

		top_3, probs = nn.make_prediction_on_single_sample(cipher_text, kb_df, model)

		st.markdown("Here are the top 3 predictions by the neural network and their associated probabilities.")

		*zip(top_3, probs)

		


