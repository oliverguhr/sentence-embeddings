import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import bert

class LabseSentenceEncoding:
    def __init__(self):        
        self.max_seq_length = 64
        self.labse_model, self.labse_layer = self.get_model(
            model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=self.max_seq_length)

        self.vocab_file = self.labse_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.labse_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = bert.bert_tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)


    def encode(self, input_text):
        input_ids, input_mask, segment_ids = self.create_input(input_text)
        return self.labse_model([input_ids, input_mask, segment_ids])
    
    def get_model(self,model_url, max_seq_length):
        labse_layer = hub.KerasLayer(model_url, trainable=False)        

        # Define input.
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                                name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")

        # LaBSE layer.
        pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

        # The embedding is l2 normalized.
        pooled_output = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

        # Define model.
        return tf.keras.Model(
                inputs=[input_word_ids, input_mask, segment_ids],
                outputs=pooled_output), labse_layer

    def create_input(self, input_strings):    
        input_ids_all, input_mask_all, segment_ids_all = [], [], []
        for input_string in input_strings:
            # Tokenize input.
            input_tokens = ["[CLS]"] + self.tokenizer.tokenize(input_string) + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            sequence_length = min(len(input_ids), self.max_seq_length)

            # Padding or truncation.
            if len(input_ids) >= self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
            else:
                input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))

            input_mask = [1] * sequence_length + [0] * (self.max_seq_length - sequence_length)

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append([0] * self.max_seq_length)

        return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)


if __name__ == "__main__":
    labse = LabseSentenceEncoding()

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    italian_sentences = ["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."]
    japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]
    german_sentences = ["Hund","Welpen sind süß.", "I genieße lange Spaziergänge mit meinem Hund am Strand."]

    english_embeddings = labse.encode(english_sentences)
    italian_embeddings = labse.encode(italian_sentences)
    japanese_embeddings = labse.encode(japanese_sentences)
    german_embeddings = labse.encode(german_sentences)

    # English-Italian similarity
    print (np.matmul(english_embeddings, np.transpose(italian_embeddings)))

    # English-Japanese similarity
    print (np.matmul(english_embeddings, np.transpose(japanese_embeddings)))

    # Italian-Japanese similarity
    print (np.matmul(italian_embeddings, np.transpose(japanese_embeddings)))

    # German-Japanese similarity
    print (np.matmul(german_embeddings, np.transpose(japanese_embeddings)))
