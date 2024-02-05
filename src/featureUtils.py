import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer
import re
PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # padding符号, bert中综合信息符号
# tokenizer = BertTokenizer.from_pretrained('bert-small-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')

stopwords = stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()


def stem(word):
    # The Stem Analyzer removes morphosyntactic affixes from words, leaving only the stem.
    return stemmer.stem(word)

def aminer12_clean_sentence(text, stemming=False):
    """
    lower -> remove punction -> remove stopwords -> stem()
    :param text:
    :param stemming:
    :return:
    """
    text = text.lower()
    for token in punct:
        text = text.replace(token, "")
    words =  re.split('-|\s',text)
    words = [w for w in words if w not in stopwords]
    # words = [ for w in words if "-" in w]
    if stemming:
        stemmed_words = []
        for w in words:
            stemmed_words.append(stem(w))
        words = stemmed_words
    return " ".join(words)


def clean_sentence(text, stemming=False):
    """
    lower -> remove punction -> remove stopwords -> stem()
    :param text:
    :param stemming:
    :return:
    """
    text = text.lower()
    for token in punct:
        text = text.replace(token, "")
    words = text.split()
    words = [w for w in words if w not in stopwords]
    # words = [ for w in words if "-" in w]
    if stemming:
        stemmed_words = []
        for w in words:
            stemmed_words.append(stem(w))
        words = stemmed_words
    return " ".join(words)


def clean_name(name):
    """
    :param name:
    :return:
    """
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
    return "_".join(x)


def transform_feature(data, f_name, k=1):
    if type(data) is str:
        data = data.split()
    assert type(data) is list
    features = []
    for d in data:
        features.append("__%s__%s" % (f_name.upper(), d))
    return features


def extract_common_features(paper):
    """
    # Get the title, abstract, venue of the paper, and clean
    :param paper:
    :return:
    """
    title = paper.get('title')
    if title:
        title = clean_sentence(title, stemming=True)
    else:
        title = ''

    abstract = paper.get('abstract')
    if abstract:
        abstract = clean_sentence(abstract, stemming=True)
    else:
        abstract = ''

    venue = paper.get('venue')
    if venue:
        venue = clean_sentence(venue, stemming=True)
    else:
        venue = ''

    features = []
    features.extend(title.split())
    features.extend(abstract.split())
    features.extend(venue.split())

    return features





def aminer12_extract_common_features(paper):
    """
    :param paper:
    :return:
    """
    title = paper.get('title')
    if title:
        title = aminer12_clean_sentence(title, stemming=True)
    else:
        title = ''

    abstract = paper.get('abstract')
    if abstract:
        abstract = aminer12_clean_sentence(abstract, stemming=True)
    else:
        abstract = ''

    venue = paper.get('venue')
    if venue:
        venue = aminer12_clean_sentence(venue, stemming=True)
    else:
        venue = ''

    features = []
    features.extend(title.split())
    features.extend(abstract.split())
    features.extend(venue.split())

    return features

def transform_textcode2(content, pad_size=512):
    token = tokenizer.tokenize(content)
    token = [CLS] + token + [SEP]

    token_ids = tokenizer.convert_tokens_to_ids(token)
    if pad_size:
        if len(token) < pad_size:
            mask_ids = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask_ids = [1] * pad_size
            token_ids = token_ids[:pad_size]

    return token_ids, mask_ids