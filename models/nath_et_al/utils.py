import spacy_udpipe
import stanza

nlp_stanza = stanza.Pipeline(lang='hy',  processors='tokenize, pos, lemma')
nlp_udpipe = spacy_udpipe.load(lang='hy')


def lemmatize(text):
    doc = nlp_stanza(text)
    return [word.lemma for sentence in doc.sentences for word in sentence.words]


def sentence_tokenizer(text):
    doc = nlp_udpipe(text)
    return [x.string for x in list(doc.sents)]
