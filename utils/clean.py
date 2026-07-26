import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (first run only)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def remove_html(text):
    """Remove HTML tags"""
    return re.sub(r"<.*?>", "", str(text))


def remove_urls(text):
    """Remove URLs"""
    return re.sub(r"http\S+|www\S+", "", text)


def remove_special_characters(text):
    """Keep only letters and spaces"""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def to_lowercase(text):
    """Convert text to lowercase"""
    return text.lower()


def remove_extra_spaces(text):
    """Remove extra spaces"""
    return " ".join(text.split())


def remove_stopwords(text):
    """Remove English stopwords"""
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def lemmatize_text(text):
    """Lemmatize words"""
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def clean_text(text):
    """
    Complete cleaning pipeline
    """
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = remove_extra_spaces(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)

    return text