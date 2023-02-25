import nltk
nltk.download('punkt')
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

nltk.download('punkt')

st = StanfordNERTagger('stanford-ner-2018-10-16\classifiers\english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner-2018-10-16\stanford-ner.jar',
                       encoding='utf-8')

text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)
print(classified_text）