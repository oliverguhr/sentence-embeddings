# LaBSE Wrapper: A simple wrapper for the "Language-Agnostic BERT Sentence Embedding" Model

This repo contains a simple wrapper for the the LaBSE language model by Feng. et al. 
You can read more about this model at [Google AI Blog](https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html) and the [Paper](https://arxiv.org/abs/2007.01852).


## Usage

1. Clone the repository and run

```
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

2. Use the model to generate sentence your multi language sentence embeddings. If you run the model for the first time, the model will be downloaded automatically. Please note that this can take a while

```
labse = LabseSentenceEncoding()

english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]    
german_sentences = ["Hund","Welpen sind süß.", "I genieße lange Spaziergänge mit meinem Hund am Strand."]

english_embeddings = labse.encode(english_sentences)
german_embeddings = labse.encode(german_sentences)
```

3. Compare the similarity of sentences in different languages

```
# German-English similarity
print (np.matmul(german_embeddings, np.transpose(english_embeddings)))

#[[0.92966205 0.34716153 0.39702922]
# [0.41559264 0.8498492  0.37241483]
# [0.44947135 0.3624616  0.94378114]]
```
