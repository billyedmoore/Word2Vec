import nltk
import numpy as np
from numpy.typing import NDArray
import random


def _load_text(filename: str) -> str:
    """
    Load the corpus from file, remove empty or commented lines then join into a big string.
    """
    with open(filename, "r") as f:
        # remove comment lines and empty lines
        lines = list(filter(lambda x: not (x.startswith("//") or not x), f.readlines()))

    return " ".join(lines)


def _tokenize_text(corpus: str) -> list[str]:
    """
    Tokenize the corpus using NLTK.
    Include only the words and convert them all to lowercase.
    """
    tokens = nltk.word_tokenize(corpus)
    tokens = filter(lambda token: all([c.isalpha() for c in token]), tokens)
    return [token.lower() for token in tokens]


def _sampling_prob(
    n_occurances: int, total_n_words_in_corpus: int, sample: float = 0.001
):
    """
    Get the probablity of a word to be sampled.
    """
    fraction_of_corpus = n_occurances / total_n_words_in_corpus
    # An L shape, removes more frequent words more aggressivly
    probablity_of_being_sampled = (
        (((fraction_of_corpus / sample) ** (0.5)) + 1) * sample / fraction_of_corpus
    )
    return probablity_of_being_sampled


def _sample(tokens: list[str]):
    """
    Take a sample of the tokens using the _sample_prob function as the weight.
    """
    freq_map = {}
    for token in tokens:
        if token in freq_map:
            freq_map[token] += 1
        else:
            freq_map[token] = 1

    total_len = len(tokens)
    prob_map = {}
    for token, freq in freq_map.items():
        prob_map[token] = _sampling_prob(freq, total_len)

    sample = []
    for token in tokens:
        p = prob_map[token]
        if random.choices([True, False], weights=[p, 1 - p], k=1)[0]:
            sample.append(token)
    return sample


def _create_token_index_maps(tokens: list[str]) -> tuple[dict, list]:
    """
    Build maps to map the index to the string and visa-versa.
    """
    token_to_index = {}
    index_to_token = []

    for i, token in enumerate(set(tokens)):
        token_to_index[token] = i
        index_to_token.append(token)

    return (token_to_index, index_to_token)


def _tokens_to_training_data(
    tokens: list[str], window_size: int, n_negative_samples: int
) -> tuple[list[str], list[str], list[int]]:
    """
    Generate a list of inputs and outputs by collecting the all tokens
    within the window of each token.
    Returns a tuple (targets,context,labels).
    Tokens are returned as strings (not one-hot-encoded).
    """
    targets = []  # inputs
    contexts = []  # outputs

    def add_value(target, context, label):
        targets.append(target)
        contexts.append(context)
        labels.append(label)

    labels = []

    valid_range = range(0, len(tokens))

    for i, token in enumerate(tokens):
        positive_values = set()
        for offset in range(1, window_size + 1):
            if i - offset in valid_range:
                context = tokens[i - offset]
                positive_values.add(context)
                add_value(token, context, 1)
            if i + offset in valid_range:
                context = tokens[i + offset]
                positive_values.add(context)
                add_value(token, context, 1)

        for _ in range(n_negative_samples):
            negative_token = random.choice(tokens)
            while negative_token in positive_values:
                negative_token = random.choice(tokens)
            add_value(token, negative_token, 0)

    return (targets, contexts, labels)


def _encode_training_data(
    unencoded_target: list[str],
    unencoded_context: list[str],
    _labels: list[int],
    token_to_index_map: dict,
):
    print(unencoded_context)
    print(unencoded_target)
    targets = np.array([token_to_index_map[token] for token in unencoded_target])
    contexts = np.array([token_to_index_map[token] for token in unencoded_context])
    labels = np.array(_labels)
    return targets, contexts, labels


def encode_token(token: str, token_to_index: dict[str, int]) -> NDArray:
    """
    One-hot encode a single token.
    """
    X = np.zeros((1, max(token_to_index.values()) + 1))
    X[0, token_to_index[token]] = 1

    return X


def file_to_training_data(
    filename: str, window_size: int = 3, subsample: bool = True
) -> tuple[NDArray, NDArray, NDArray, list[str], dict[str, int]]:
    text = _load_text(filename)
    tokens = _tokenize_text(text)
    if subsample:
        tokens = _sample(tokens)

    print(tokens)
    (token_to_index, index_to_token) = _create_token_index_maps(tokens)
    (unencoded_targets, unencoded_contexts, labels) = _tokens_to_training_data(
        tokens, window_size, 5
    )
    (targets, contexts, labels) = _encode_training_data(
        unencoded_targets, unencoded_contexts, labels, token_to_index
    )
    return targets, contexts, labels, index_to_token, token_to_index
