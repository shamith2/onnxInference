import argparse
import os
import sys

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from babyLlama import babyLlama
from babyDecoder import babyDecoder

def test(prompt: str) -> str:
    from transformers import AutoModelForCausalLM, GenerationConfig

    model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(input_ids, max_length = 1000, num_beams=1)

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return output_text

def inference(model_name:str, prompt: str, temperature: float = 1.0) -> str:
    # load tokenizer
    tokenizer_file = os.path.join('tinyStories', 'tinyTokenizer.json')
    tokenizer = Tokenizer.from_file(tokenizer_file)
    
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # load checkpoint
    if model_name == 'babyLlama':
        checkpoint = torch.load(os.path.join('babyLlama', '5-epoch-4.0M-checkpoint-4-0.432.pt'))
    
    elif model_name == 'babyDecoder':
        checkpoint = torch.load(os.path.join('babyDecoder', '5-epoch-28.0M-checkpoint-4-0.394.pt'))
    
    else:
        sys.exit()

    model_args = checkpoint['model_args']
    model_args.batch_size = 1

    print(model_args)

    print("using device: {}".format(model_args.device))

    if model_name == 'babyLlama':
        model = babyLlama(model_args).to(model_args.device)
    
    elif model_name == 'babyDecoder':
        model = babyDecoder(model_args).to(model_args.device)
    
    else:
        sys.exit()

    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        model.eval()

        tokens = tokenizer.encode(prompt) # , return_tensors='pt')

        input_tokens = torch.tensor(tokens.ids, dtype=torch.int, device=model_args.device).unsqueeze(0)
        # input_tokens = tokens.to(model_args.device)

        output_tokens = model.generate(input_tokens, eos_token_id=2, temperature=temperature)

    generated_text = tokenizer.decode(output_tokens.squeeze(0).tolist())

    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference of models trained on tinyStories dataset')
    parser.add_argument('--model', type=str, help='model',
                        default='babyDecoder')
    parser.add_argument('--prompt', type=str, help='prompt',
                        default=None)

    args = parser.parse_args()

    prompt = ('Once upon a time there was a little girl named Lucy. She was very adventurous. She loved to explore the world around her, especially when it was bright and sunny outside.\nOne day, while exploring the nearby park, Lucy came across a ladder leaning on a wall. She was curious to see what\'s on top, so she climbed the ladder, but when she reached the top, the ladder fell and she was stuck.\nA nearby park ranger noticed her and shouted out, "')

    if not args.prompt:
        args.prompt = prompt

    generated_text = inference(args.model, args.prompt, temperature=0)

    print(generated_text)

    # test_gen_text = test(args.prompt)

    # print(test_gen_text)
