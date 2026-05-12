from transformers import BertTokenizer, BertModel
import torch

task = 'pursuit'
scenes = ['hunting', 'finishing', 'collaboration', 'robustness'] 

tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
model = BertModel.from_pretrained("bert-large-cased-whole-word-masking")

for scene in scenes:
    for i in range(50):
        with open(f'/data/situationDesc/{task}/{scene}/{i}.txt', 'r') as f:
            text = f.readlines()
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)

            torch.save(output.pooler_output, f"/data/situationEmb/{task}/{scene}/{i}.pt")

