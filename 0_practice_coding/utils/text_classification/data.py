import re
from functools import partial

# To import stopwords, we must download nltk stopword package.
# ```
# import nltk
# nltk.download('stopwords') 
# ```
from nltk.corpus import stopwords
from keras.utils import to_categorical


stop_words = stopwords.words('english')
# [
#     'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
#     'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
#     'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
#     'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
#     'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
#     'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
#     'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
#     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
#     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
#     'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
#     'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
#     'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
#     'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
#     'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
#     'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
#     'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
#     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
#     'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
#     'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
#     'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
#     'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
# ]


def clean_string(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove urls
    text = re.sub(r'http\S+', ' ', text)
    # Remove mentions
    text = re.sub(r'@\w+', ' ', text)
    # Remove hastags
    text = re.sub(r'#\w+', ' ', text)
    # Remove digits
    text = re.sub(r'\d+', ' ', text)
    # Remove html tags
    text = re.sub('r<.*?>',' ', text)
    # Remove stop_words
    text = text.split()
    text = " ".join([word for word in text if not word in stop_words])

    return text


def train_test_split_by_df(df, test_size):
    # Shuffle df
    df = df.sample(frac=1, random_state=204)

    # Split train - test set
    train_df = df.iloc[:int((1 - test_size) * len(df))]
    test_df = df.iloc[int((1 - test_size) * len(df)):]

    return train_df, test_df


def prepare_onehot_label(df, label_col_name, index_label_dict):
    # From string label to index label
    df[f'{label_col_name}_index'] = df[label_col_name].replace(index_label_dict)

    # From index label to onehot label
    to_categorical_n_classes_fn = partial(to_categorical, num_classes=len(index_label_dict))
    df[f'{label_col_name}_onehot'] = df[f'{label_col_name}_index'].apply(to_categorical_n_classes_fn)

    return df
