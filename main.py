import nltk

def get_corpus(filename: str) -> str:
    """
    Load the corpus from file, remove empty or commented lines then join into a big string.
    """
    with open(filename,"r") as f:
        # remove comment lines and empty lines
        lines = list(filter(lambda x: not (x.startswith("//") or not x),f.readlines()))

    return " ".join(lines)


def tokenize_corpus(corpus: str) -> list[str]:
    """
    Tokenize the corpus using NLTK.
    Include only the words and convert them all to lowercase.
    """
    tokens = (nltk.word_tokenize(corpus))
    tokens = filter(lambda token: all([c.isalpha() for c in token]),tokens)
    return [token.lower() for token in tokens]

def build_index(tokens: list[str]) -> tuple[dict,list]:
    """
    Build maps to map the index to the string and visa-versa.
    """
    token_to_index = {}
    index_to_token = []

    for i,token in enumerate(set(tokens)):
        token_to_index[token] = i
        index_to_token.append(token)

    return (token_to_index,index_to_token)

def generate_training_data(tokens: list[str],window_size: int) -> tuple[list[str],list[str]]:
    """
    Generate a list of inputs and outputs by collecting the all tokens within the window of each token.
    Response is a tuple (X,Y) or in natural language (inputs,outputs). 
    Tokens are returned as strings (not one-hot-encoded).
    """
    X = [] # inputs
    Y = [] # outputs

    valid_range = range(0,len(tokens))

    for i,token in enumerate(tokens):
        for offset in range(1,window_size+1):
            if i-offset in valid_range:
                X.append(token)
                Y.append(tokens[i-offset])
            if i+offset in valid_range:
                X.append(token)
                Y.append(tokens[i+offset])


if __name__ == "__main__":
    print(tokenize_corpus(get_corpus("tomorrow_and_tomorrow_and_tomorrow.txt")))


