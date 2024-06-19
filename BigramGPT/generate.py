import torch
from BigramGPT import GPTLanguageModel, decode
import argparse

def parse_model_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.pth', help='path to the model file')
    parser.add_argument('--length', type=int, default=100, help='number of tokens to generate')
    args = parser.parse_args()
    return args.model_path, args.length

if __name__ == '__main__':
    model = GPTLanguageModel()
    model_path, length = parse_model_path()
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = model.to(device)
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=length)[0].tolist()))
    
    
    