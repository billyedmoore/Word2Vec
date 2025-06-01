import keras
from numpy.typing import NDArray

def create_model(number_of_unique_tokens: int,vector_length: int):
    """
    Create a new model (not yet an embedding model) doesn't train the model.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(number_of_unique_tokens,)),
        keras.layers.Dense(vector_length),
        keras.layers.Dense(number_of_unique_tokens,activation="relu"),
    ])
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    model.summary()
    return model

def train_model(model: keras.Model,X: NDArray,Y: NDArray, epochs: int=25,batch_size: int=32):
    """
    Train the model using all the data.
    """
    # `1` is the correct value for verbose but pyright doesnt agree
    model.fit(X,Y,epochs=epochs,batch_size=batch_size,verbose=1) #  type:ignore 

def convert_to_embedding_model(model: keras.Model):
    """
    Pop the top layer.
    """
    model.pop()
    model.summary()

def upsert(model: keras.Model, X: NDArray):
    """
    Run a single token against the embedding model and return the embedding.
    """
    return model(X).cpu().detach().numpy() 



