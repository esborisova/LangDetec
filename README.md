# LangDetec

Pipeline for training an ML classification model for document language detection. 

# Dataset

European Parliament Proceedings Parallel Corpus (EPPPC) 1996-2011 (Koehn 2005). 

The data corpus can be downloaded [here](https://www.statmt.org/europarl/).

# Approach

Multiclass supervised classification based on TF-IDF weighted N-character-gram text representation.

# Train and test corpus

The folowing langauges were selected:

- English ('en')
- Danish ('da')
- German ('de')
- Swedish ('sv')
- Italian ('it'). 

# ML algorithm

Multinomial Naive Bayes classifier.
