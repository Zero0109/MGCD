import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from copy import deepcopy
from sklearn.metrics import r2_score
import random
import os
from tqdm import tqdm

from Dataset import TrainDataset, FinetuningDataset, TestDataset
from MGCE import MGCE

def train(state,
          llm,
          comm_name,
          positive,
          dataType,
          task,
          trainType,
          testType,
          sceneType,
          algorithm_num,
          scene_totnum,
          scene_pernum,
          capability_num,
          terrain_emb_dir,
          rule_emb_dir,
          situation_emb_dir,
          gpu_id=0,
          num_workers=4,
          batch_size=32,
          train_epochs=500,
          finetuning_epochs=500):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    traindataset = TrainDataset(positive, dataType, device, task=task, _type=trainType, algorithm_num=algorithm_num,
                                scene_num=scene_totnum)
    traindata_loader = data.DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    Q = torch.zeros((scene_totnum, capability_num)).to(device)
    for i in range(scene_totnum):
        Q[i, i // scene_pernum] = 1

    terrain_embs = []
    rule_embs = []
    situation_embs = []
    for i in range(len(sceneType)):
        for j in range(scene_pernum):
            terrain_embs.append(torch.load(terrain_emb_dir + f'{sceneType[i]}/{j}.pt').reshape((-1,)).tolist())
            rule_embs.append(torch.load(rule_emb_dir + f'{sceneType[i]}/{j}.pt').reshape((-1,)).tolist())
            situation_embs.append(torch.load(situation_emb_dir + f'{sceneType[i]}/{j}.pt').reshape((-1,)).tolist())
    terrain_embs = torch.tensor(terrain_embs, device=device)
    rule_embs = torch.tensor(rule_embs, device=device)
    situation_embs = torch.tensor(situation_embs, device=device)

    mgce = MGCE(t=algorithm_num,
                m=scene_totnum,
                d=capability_num,
                Q=Q,
                terrain_embs=terrain_embs,
                rule_embs=rule_embs,
                situation_embs=situation_embs,
                terrain_dim = 1024,
                rule_dim = 1024,
                situation_dim = 1024,
                embed_dim = 256,
                attr_hidden_dim = 64,
                tpl_hidden_dim = 32,
                num_heads = 8,
                num_layers = 2,
                dropout = 0.1)
    mgce = mgce.to(device)
    optimizer = torch.optim.Adam(mgce.parameters(), lr=0.001, weight_decay=1e-5)
    loss_function = nn.MSELoss(reduction='mean')

    if state == "train":
        best_loss = float('inf')
        loop = tqdm(range(train_epochs))
        for epoch in enumerate(loop):
            running_loss = 0.0
            for i, train_data in enumerate(traindata_loader):
                optimizer.zero_grad()
                pair, y = train_data
                y_hat = mgce(pair[:, 0], pair[:, 1], state=0)

                loss = loss_function(y_hat, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                loop.set_postfix({
                    'loss': f'{loss.item():.8f}',
                    'avg': f'{running_loss / (i + 1):.8f}'
                })
            avg_loss = running_loss / len(traindata_loader) if len(traindata_loader) > 0 else running_loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_param = deepcopy(mgce.state_dict())
        capabilities = get_capabilities(deepcopy(mgce), algorithm_num)
        score = np.exp(np.mean(np.log(capabilities), axis=1))
        #max_algo_ind = np.argmax(score)
        torch.save(best_param,f'/checkpoints/{llm}/{task}/{comm_name}_weights.pt')
        return

    elif state == "fine":
        best_param = torch.load(f"checkpoints/{llm}/{task}/{comm_name}_weights.pt", map_location=f"cuda:{gpu_id}")
        for max_algo_ind in range(algorithm_num):
            mgce.load_state_dict(best_param)
            for param in mgce.parameters():
                param.requires_grad = False
            for module in [
                mgce.g_mlp,
                mgce.situation_proj,
                mgce.gate_situation,
                mgce.fusion_fc
            ]:
                for p in module.parameters():
                    p.requires_grad = True

            finetuningdataset = FinetuningDataset(positive, dataType, device, task=task, _type=testType,
                                                algorithm_ind=max_algo_ind, scene_num=scene_totnum)
            finetuningdata_loader = data.DataLoader(finetuningdataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers)
            testdataset = TestDataset(positive, dataType, device, task=task, _type=testType, algorithm_ind=max_algo_ind,
                                    algorithm_num=algorithm_num, scene_num=scene_totnum)
            testdata_loader = data.DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, mgce.parameters()), lr=0.0001, weight_decay=1e-5)
            for epoch in range(finetuning_epochs):
                for i, finetuning_data in enumerate(finetuningdata_loader):
                    optimizer2.zero_grad()
                    pair, y = finetuning_data
                    y_hat = mgce(pair[:, 0], pair[:, 1], state=1)
                    loss = loss_function(y_hat, y)
                    loss.backward()
                    optimizer2.step()
                    running_loss += loss.item()

            y_hat_list, y_list = test(deepcopy(mgce), testdata_loader)

            os.makedirs(f'results/{task}/MGCE/{llm}', exist_ok=True)
            with open(
                    f'results/{task}/MGCE/{llm}/{comm_name}_testType_{testType}_alg_ind_{max_algo_ind}.txt',
                    'w') as f:
                for i in range(len(y_hat_list)):
                    f.write(f'{y_hat_list[i]} {y_list[i]}\n')


def get_capabilities(model, algorithm_num):
    model.eval()
    capabilities = []
    for i in range(algorithm_num):
        with torch.no_grad():
            capa = model.get_capabilities(torch.tensor(i).to(torch.device((f'cuda:{gpu_id}') if torch.cuda.is_available() else 'cpu'))).detach().cpu().tolist()
            capabilities.append(capa)
    model.train()
    return np.array(capabilities)

if __name__ == "__main__":
    positive = True
    state = "train"
    llm = "gpt5"
    dataType = 'ts'
    scene_pernum = 50
    gpu_id = 0
    num_workers = 0
    batch_size = 32
    train_epochs = 500
    finetuning_epochs = 500

    task = 'pursuit'
    trainType = 'random'
    testType = 'obstacle2'
    algorithm_num = 6
    scene_totnum = 200
    capability_num = 4
    sceneType = ['hunting', 'finishing', 'collaboration', 'robustness']
    terrain_emb_dir = f'/data/terrainEmb/'
    rule_emb_dir = f'/data/ruleEmb/'
    situation_emb_dir = f'/data/situationEmb/'

    comm_name = f'alg_num_{algorithm_num}_sce_num_{scene_totnum}_dataType_{dataType}_trainType_{trainType}'
    train(state=state,
            llm=llm,
            comm_name=comm_name,
            positive=positive,
            dataType=dataType,
            task=task,
            trainType=trainType,
            testType=None,
            sceneType=sceneType,
            algorithm_num=algorithm_num,
            scene_totnum=scene_totnum,
            scene_pernum=scene_pernum,
            capability_num=capability_num,
            terrain_emb_dir=terrain_emb_dir,
            rule_emb_dir=rule_emb_dir,
            situation_emb_dir=situation_emb_dir,
            gpu_id=gpu_id,
            num_workers=num_workers,
            batch_size=batch_size,
            train_epochs=train_epochs,
            finetuning_epochs=finetuning_epochs)
