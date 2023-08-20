# Hive Board Game Ngram Language Model

**Note:** This project is based on the principles outlined in the book "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. You can find more details in [Chapter 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf) of the book.

The Hive Board Game Ngram Language Model is a probabilistic model specifically designed for generating and evaluating sequences of moves in the Hive board game. It utilizes Ngram modeling to estimate the probabilities of different move sequences and supports tasks such as move generation and strategy analysis.

## Overview

This implementation provides an `NgramModel` class that has been adapted for the Hive board game, enabling the following functionalities:
- Ngram modeling with adjustable N values.
- Estimation of probabilities for sequences of moves.
- Calculation of perplexity for evaluating move sequences.
- Random move generation based on historical context.
