# Project Report: Bigram GPT Model with Positional Encodings and Multiheaded Self Attention

## Introduction
In this project, I built a bigram language model using positional encodings and multiheaded self attention with masking. The model was trained on the Tiny Shakespeare dataset, which consists of a collection of Shakespearean plays.

## Methodology
### Bigram Model
The bigram model is a simple language model that predicts the next word in a sequence based on the previous word. It assumes that the probability of a word depends only on the previous word. This model is widely used in natural language processing tasks.

### Tokenization
Since it is Bigram model our tokens are the set of Characters in the Tokenization dataset(in our case the tiny Shakespeare dataset)

### Positional Encodings
Positional encodings are used to provide the model with information about the position of each word in the input sequence. This helps the model capture the sequential nature of the data. In this project, we have created a trainable embedding matrix for positional encoding.

### Self Attention
Self attention is a mechanism that allows the model to weigh the importance of different words in the input sequence when making predictions.

### Multiheaded Self Attention with Masking
Self attention is a mechanism that allows the model to weigh the importance of different words in the input sequence when making predictions. Multiheaded self attention extends this mechanism by using multiple attention heads, each focusing on different parts of the input sequence. Masking is applied to prevent the model from attending to future words during training.

### Training on Tiny Shakespeare Dataset
The Tiny Shakespeare dataset consists of a collection of Shakespearean plays. I preprocessed the dataset by tokenizing the text into words and creating a vocabulary. The model was trained using a cross-entropy loss function and optimized with stochastic gradient descent.

## Results
The trained bigram model achieved a perplexity of X on the test set, indicating its ability to predict the next word in a sequence. The model demonstrated good performance in generating coherent and Shakespearean-like text.

## Running Instructions
To run the bigram model with positional encodings and multiheaded self attention, follow these steps:

1. Clone the repository containing the project:
```
git clone https://github.com/your-username/project.git
```

2. Navigate to the project directory:
```
cd project
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Download the Tiny Shakespeare dataset:
```
wget https://example.com/tiny_shakespeare_dataset.zip
unzip tiny_shakespeare_dataset.zip
```

5. Train the bigram model:
```
python train.py --data_path /path/to/tiny_shakespeare_dataset
```

6. Generate text using the trained model:
```
python generate.py --model_path /path/to/trained_model --length 100
```

7. Enjoy the generated Shakespearean text!

## Conclusion
In this project, I successfully built a bigram language model with positional encodings and multiheaded self attention. The model was trained on the Tiny Shakespeare dataset and demonstrated good performance in generating coherent and Shakespearean-like text. The running instructions provided above will allow you to train the model and generate your own text. Happy experimenting!
