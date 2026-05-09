import torch
import torch.nn as nn
import torch.nn.functional as F

class MonotoneLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_raw = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight_raw)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.weight_raw)
        return F.linear(x, w, self.bias)


class MonotoneMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        act1 = nn.Tanh()
        act2 = nn.Sigmoid()
        self.net = nn.Sequential(
            MonotoneLinear(in_dim, hidden_dim),
            act1,
            MonotoneLinear(hidden_dim, out_dim),
            act2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MGCE(nn.Module):
    def __init__(self, t, m, d, Q, terrain_embs, rule_embs, situation_embs, terrain_dim,
                 rule_dim, situation_dim, embed_dim, attr_hidden_dim, tpl_hidden_dim,
                 num_heads, num_layers, dropout):
        super(MGCE, self).__init__()

        self.terrain_proj = nn.Linear(terrain_dim, embed_dim)
        self.rule_proj = nn.Linear(rule_dim, embed_dim)
        self.situation_proj = nn.Linear(situation_dim, embed_dim)
        self.modality_emb = nn.Parameter(torch.randn(3, embed_dim))
        self.gate_terrain = nn.Linear(embed_dim, 1)
        self.gate_rule = nn.Linear(embed_dim, 1)
        self.gate_situation = nn.Linear(embed_dim, 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4,
                                                   dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion_fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, d),
            nn.Sigmoid()
        )

        self.Q = Q
        self.terrain_embs = terrain_embs
        self.rule_embs = rule_embs
        self.situation_embs = situation_embs

        self.algo_hidden = nn.Embedding(t, attr_hidden_dim)
        self.scene_hidden = nn.Embedding(m, attr_hidden_dim)
        self.capability_mlp = nn.Sequential(
            nn.Linear(attr_hidden_dim, attr_hidden_dim),
            nn.LayerNorm(attr_hidden_dim),
            nn.ReLU(),
            nn.Linear(attr_hidden_dim, d)
        )
        self.diff_mlp = nn.Sequential(
            nn.Linear(attr_hidden_dim, attr_hidden_dim),
            nn.LayerNorm(attr_hidden_dim),
            nn.ReLU(),
            nn.Linear(attr_hidden_dim, d)
        )
        self.discr_mlp = nn.Sequential(
            nn.Linear(attr_hidden_dim, attr_hidden_dim),
            nn.LayerNorm(attr_hidden_dim),
            nn.ReLU(),
            nn.Linear(attr_hidden_dim, 1)
        )
        self.floor_mlp = nn.Sequential(
            nn.Linear(attr_hidden_dim, attr_hidden_dim),
            nn.LayerNorm(attr_hidden_dim),
            nn.ReLU(),
            nn.Linear(attr_hidden_dim, 1)
        )

        self.f_mlp = MonotoneMLP(in_dim=d, hidden_dim=tpl_hidden_dim, out_dim=1)
        self.g_mlp = nn.Sequential(
            nn.Linear(d, tpl_hidden_dim),
            nn.ReLU(),
            nn.Linear(tpl_hidden_dim, d)
        )

        self.tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, algorithm_id, scene_id, state):
        t = self.terrain_proj(self.terrain_embs[scene_id])
        r = self.rule_proj(self.rule_embs[scene_id])
        s = self.situation_proj(self.situation_embs[scene_id])
        gates = torch.cat([
            self.gate_terrain(t),
            self.gate_rule(r),
            self.gate_situation(s)
        ], dim=1)
        weights = self.softmax(gates).unsqueeze(-1)
        x = torch.stack([t, r, s], dim=1)
        x = x + self.modality_emb.unsqueeze(0)
        x = x * weights
        x_t = x.transpose(0, 1)
        x_t = self.transformer(x_t)
        r_sce = x_t.mean(dim=0)
        q2 = self.fusion_fc(r_sce)

        capabilities = self.sigmoid(self.capability_mlp(self.algo_hidden(algorithm_id)))
        difficulties = self.sigmoid(self.diff_mlp(self.scene_hidden(scene_id)))
        discriminations = nn.functional.softplus(self.discr_mlp(self.scene_hidden(scene_id)))
        chance_floors = 0.2 * self.sigmoid(self.floor_mlp(self.scene_hidden(scene_id)))

        q1 = self.Q[scene_id]
        q = self.softmax(0.9*q1 + 0.1*q2)

        assert state in [0, 1]
        if state == 0:
            y = chance_floors + (1 - chance_floors) * self.f_mlp(q * (capabilities - difficulties) * discriminations)
        elif state == 1:
            y = chance_floors + (1 - chance_floors) * self.f_mlp(self.g_mlp(q * (capabilities - difficulties) * discriminations))

        return y

    def get_capabilities(self, algorithm_id):
        capabilities = self.sigmoid(self.capability_mlp(self.algo_hidden(algorithm_id)))
        return capabilities.data

    def get_attributes(self, scene_id):
        difficulties = self.sigmoid(self.diff_mlp(self.scene_hidden(scene_id)))
        discriminations = nn.functional.softplus(self.discr_mlp(self.scene_hidden(scene_id)))
        chance_floors = 0.2 * self.sigmoid(self.floor_mlp(self.scene_hidden(scene_id)))
        return difficulties.data, discriminations.data, chance_floors.data