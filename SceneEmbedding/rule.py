from transformers import BertTokenizer, BertModel
import numpy as np
import torch

hunting_prompt = "The area size is 20m x 10m; the vehicle's vertical projection area is 0.6m x 0.5m; the maximum speed of pursuer is 0.8 m/s; the maximum speed of evader is 0.8 m/s; the time limit is 10 time steps."
finishing_prompt = "The area size is 20m x 10m; the vehicle's vertical projection area is 0.6m x 0.5m; the maximum speed of pursuer is 0.8 m/s; the maximum speed of evader is 0.8 m/s; the time limit is 5 time steps."
collaboration_prompt = "The area size is 20m x 10m; the vehicle's vertical projection area is 0.6m x 0.5m; the maximum speed of pursuer is 0.8 m/s; the maximum speed of evader is 0.8 m/s; the time limit is 10 time steps. The pursuers have constant awareness of the evader's position."
robustness_prompt = ["The area size is 20m x 10m; the vehicle's vertical projection area is 0.6m x 0.5m; the maximum speed of pursuer is 0.8 m/s; the maximum speed of evader is 0.8 m/s; the time limit is 10 time steps. One of the pursuers is unable to detect the evader through its camera.",
                     "The area size is 20m x 10m; the vehicle's vertical projection area is 0.6m x 0.5m; two of the pursuers have a maximum speed of 0.8 m/s, while one pursuer is limited to a maximum speed of 0.4 m/s; the maximum speed of evader is 0.8 m/s; the time limit is 10 time steps.",
                     "The area size is 20m x 10m; the vehicle's vertical projection area is 0.6m x 0.5m; two of the pursuers have a maximum speed of 0.8 m/s, while one pursuer can reach up to 1.2 m/s; the maximum speed of evader is 0.8 m/s; the time limit is 10 time steps.",
                     "The area size is 20m x 10m; the vehicle's vertical projection area is 0.6m x 0.5m; the maximum speed of pursuer is 0.8 m/s; the maximum speed of evader is 1.2 m/s; the time limit is 10 time steps.",
                     "The area size is 20m x 10m; the vehicle's vertical projection area is 0.6m x 0.5m; the maximum speed of pursuer is 0.8 m/s; the maximum speed of evader is 1.6 m/s; the time limit is 10 time steps."]

spread_prompt = "The area size is 4 x 4; the radius of agent is 0.08, the radius of landmark is 0.04; the maximum speed of agent is 1; the time limit is 50 time steps"
speaker_listener_prompt = "The area size is 4 x 4; the radius of agent is 0.075, the radius of landmark is 0.04; the maximum speed of agent is 1; the time limit is 50 time steps"
tag_prompt = "The area size is 4 x 4; the radius of prey is 0.05, the radius of predator is 0.075; the maximum speed of prey is 1.3, the maximum speed of predator is 1; the time limit is 25 time steps"


constraint = np.loadtxt("/data/constraint.txt", dtype=int)

tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
model = BertModel.from_pretrained("bert-large-cased-whole-word-masking")

task = 'pursuit'
scenes = ['hunting', 'finishing', 'collaboration', 'robustness']

for scene in scenes:
    for i in range(50):

        if scene == "robustness":
            prompt = robustness_prompt[constraint[i][0]]
        else:
            prompt = eval(f"{scene}_prompt")

        encoded_input = tokenizer(prompt, return_tensors='pt')
        output = model(**encoded_input)

        torch.save(output.pooler_output, f"/data/ruleEmb/{task}/{scene}/{i}.pt")