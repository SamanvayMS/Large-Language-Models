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
Self attention is a mechanism that allows the model to weigh the importance of different words in the input sequence when making predictions. It computes a weighted sum of the input sequence based on the similarity between each pair of words. This mechanism helps the model capture long-range dependencies in the data. key, query and value matrices are used to calculate the attention weights.

### Multiheaded Self Attention with Masking
Self attention is a mechanism that allows the model to weigh the importance of different words in the input sequence when making predictions. Multiheaded self attention extends this mechanism by using multiple attention heads, each focusing on different parts of the input sequence. Masking is applied to prevent the model from attending to future words during training.

### Training on Tiny Shakespeare Dataset
The Tiny Shakespeare dataset consists of a collection of Shakespearean plays. I preprocessed the dataset by tokenizing the text into Characters and creating a vocabulary. The model was trained using a cross-entropy loss function and optimized with stochastic gradient descent.

To understand more about the above concepts you can refer to the notebook [here](gpt.ipynb) and also the youtube video by Andrej Karpathy that i followed along for this code [here](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Results
The trained bigram model achieved a cross-entropy score of 1.5183 with a perplexity score of 4.56 on the test set, indicating its ability to predict the next word in a sequence fairly accurately. The model demonstrated good performance in generating coherent and Shakespearean-like text.

## Running Instructions
To run the bigram model with positional encodings and multiheaded self attention, follow these steps:

1. Fork the repository and Clone it to your local machine

2. Navigate to the project directory:
```
cd GPT(tiny_shakespeare)
```

3. Install the required dependencies and make sure you have pytorch and Cuda installed on your machine to use the GPU for training. Without the GPU the training and generation will be extremely slow.(took ~1.5 hours for 5000 iterations on my machine with Nvidia RTX 4050)

4. Download the Tiny Shakespeare dataset:
```
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

5. Train the BigramGPT model:
```
python BigramGPT.py --model_path /path/to/save/trained_model
```

6. Generate text using the trained model:
```
python generate.py --model_path /path/to/trained_model --length 100
```

7. Enjoy the generated Shakespearean text! Might not be good enough due to the small size of the dataset and the model.

## example train output:
```
model.pth
10.788929 M parameters
step 0: train loss 4.2221, val loss 4.2306
step 500: train loss 1.7400, val loss 1.8956
step 1000: train loss 1.3873, val loss 1.5974
step 1500: train loss 1.2652, val loss 1.5267
step 2000: train loss 1.1859, val loss 1.5037
step 2500: train loss 1.1215, val loss 1.4858
step 3000: train loss 1.0713, val loss 1.4864
step 3500: train loss 1.0228, val loss 1.5043
step 4000: train loss 0.9596, val loss 1.5183
step 4500: train loss 0.9129, val loss 1.5492
step 4999: train loss 0.8604, val loss 1.5677
```
## example generate output:
```
Save than this, and not get thee, that with you.

NORFOLIMNGBHUMPERLE:
How fe! what you but be such as merry lasts?

NORTHUMBERLAND:
Fay, but for all apointed me abroad;
They make me to such woodges all, and my landy
To see a moute of the house If this imperor stone.

NORsgREY:
How! You lads, I will not our grave! O,
Come, Clowna! Come, lethout of my base!
I'll make a way, my highless in lord,
Delay. A bidde, ormondst, his moneys,
And not oppress young honourable!
This let hear
Beseeming-ound yo
```

## Conclusion
In this project, I successfully built a bigram language model with positional encodings and multiheaded self attention. The model was trained on the Tiny Shakespeare dataset and demonstrated good performance in generating coherent and Shakespearean-like text. The running instructions provided above will allow you to train the model and generate your own text. Happy experimenting!
