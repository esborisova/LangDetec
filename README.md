# LangDetec

Pipeline for training an ML model for document language detection. 

# Dataset

European Parliament Proceedings Parallel Corpus (EPPPC) 1996-2011 (Koehn 2005). 

The data corpus can be downloaded [her](https://www.statmt.org/europarl/).

# Approach

Multiclass supervised classification based on TF-IDF weighted N-character-gram text representation.

# Train and test corpus

English, Danish, German, Swedish and Italian document.

# ML model

Multinomial Naive Bayes classifier.
