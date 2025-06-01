from keras import Model
from numpy.typing import NDArray
from prepare_data import file_to_training_data
from model import create_model, train_model, upsert

import random


def upsert_token(model: Model, token: str, token_to_index: dict[str, int]) -> NDArray:
    """
    Vectorize a single token.
    """
    if token not in token_to_index.keys():
        raise ValueError("Token not in dictionary.")

    return upsert(model, token, token_to_index)


if __name__ == "__main__":
    targets, context, labels, tokens, token_to_index = file_to_training_data(
        "tomorrow_and_tomorrow_and_tomorrow.txt"
    )
    model = create_model(len(tokens), 30)
    train_model(model, targets, context, labels)

    for _ in range(5):
        token_to_vectorize = random.choice(tokens)
        vect = upsert_token(model, token_to_vectorize, token_to_index)
        print(f"{token_to_vectorize} -> ", vect)
