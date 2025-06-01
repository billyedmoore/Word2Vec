from keras import Model
from numpy.typing import NDArray
from prepare_data import file_to_training_data, encode_token
from model import convert_to_embedding_model, create_model, train_model, upsert

import random 


def upsert_token(model: Model, token: str,token_to_index: dict[str,int]) -> NDArray:
    """
    Vectorize a single token.
    """
    if token not in token_to_index.keys():
        raise ValueError

    encoded_token = encode_token(token,token_to_index) 
    return upsert(model,encoded_token)


if __name__ == "__main__":
    X,Y,tokens,token_to_index = file_to_training_data("data/AdventuresInWonderland.txt")
    model = create_model(len(tokens),30)
    train_model(model,X,Y)
    convert_to_embedding_model(model)
    
    for _ in range(5):
        token_to_vectorize  = random.choice(tokens)
        vect = upsert_token(model,token_to_vectorize,token_to_index)
        print(f"{token_to_vectorize} -> ",vect)

