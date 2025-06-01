import keras
import torch
from numpy import number
from numpy.typing import NDArray


def create_model(number_of_unique_tokens: int, vector_length: int):
    """
    Create a new model (not yet an embedding model) doesn't train the model.
    """

    # Model sould have two inputs target_index, context_index and one output the label (0 negative and 1 positive)
    target_input = keras.layers.Input(shape=(1,))
    context_input = keras.layers.Input(shape=(1,))

    embedding_layer = keras.layers.Embedding(
        input_dim=number_of_unique_tokens,
        output_dim=vector_length,
        name="embedding_layer",
    )

    target_vector = embedding_layer(target_input)
    context_vector = embedding_layer(context_input)

    target_vector = keras.layers.Reshape((vector_length,))(target_vector)
    context_vector = keras.layers.Reshape((vector_length,))(context_vector)

    # Measure of similarity of the vectors.
    dot_product_score = keras.layers.Dot(axes=1)([target_vector, context_vector])

    output = keras.layers.Activation("sigmoid")(dot_product_score)

    model = keras.Model(inputs=[target_input, context_input], outputs=output)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model


def train_model(
    model: keras.Model,
    target: NDArray,
    context: NDArray,
    labels: NDArray,
    epochs: int = 25,
    batch_size: int = 32,
):
    """
    Train the model using all the data.
    """
    # `1` is the correct value for verbose but pyright doesnt agree
    model.fit(
        [target, context], labels, epochs=epochs, batch_size=batch_size, verbose=1
    )  #  type:ignore


def upsert(model: keras.Model, token: str, token_to_index: dict[str, int]):
    """
    Run a single token against the embedding model and return the embedding.
    """
    return model.get_layer("embedding_layer").get_weights()[0][token_to_index[token]]
