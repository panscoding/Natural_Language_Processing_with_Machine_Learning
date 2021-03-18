import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Convert a list of text strings into word sequences
    def tokenize_text_corpus(self, texts):
        # CODE HERE
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences

def main():
    main()

if __name__ == "__main__":
    vocab_size = 100
    embedding_dim = 32
    model = EmbeddingModel(vocab_size, embedding_dim)
    texts = ['bob ate apples, and pears', 'fred ate apples!']
    sequences = model.tokenize_text_corpus(texts)
    print("sequences: ", sequences)
    print("end")