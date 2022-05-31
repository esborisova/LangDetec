# :speaking_head: LangDetec

Pipeline for training an ML classification model for document language detection. 

# Dataset

European Parliament Proceedings Parallel Corpus (EPPPC) 1996-2011 (Koehn 2005). 

The data corpus can be downloaded [here](https://www.statmt.org/europarl/).

# Approach

Multiclass supervised classification based on TF-IDF weighted N-character-grams.

# Train and test corpus

The folowing languages were selected:

:uk: English ('en')

:denmark: Danish ('da')

:de: German ('de')

:sweden: Swedish ('sv')

:it: Italian ('it'). 

# ML algorithm

Multinomial Naive Bayes classifier.

# References 
Koehn Philipp. 2005. Europarl: A Parallel Corpus for Statistical Machine Translation. *In Proceedings of Machine Translation Summit X: Papers*. 79–86. 13–15 September. Phuket.
