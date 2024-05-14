import os.path
import numpy as np
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from typing import Any, Optional
from torch_geometric.nn import TransformerConv, Sequential, LayerNorm
from torch_geometric.nn.inits import glorot, zeros

from .base_learner import BaseLearner
from .metric import CosineSimilarity 
from .processor import KNNSparsify, NonLinearize, Symmetrize, Normalize
from .utils import knn_fast
from .attention_learner import AttLearner
from .mlp_learner import MLPLearner
import torch.nn.init as init
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt


from torchmetrics.classification.accuracy import Accuracy


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_dropout=False, dropout_p=0.5):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.linear1 = nn.Linear(in_channels, mid_channels)
        self.batch_norm1 = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(mid_channels, out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        
        self.identity_layer = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.linear2(out)
        out = self.batch_norm2(out)
        
        identity = self.identity_layer(identity)
        
        out += identity
        out = self.relu(out)
        return out


class MLP_NEW(LightningModule):
    r"""A MLP with two linear layers.

    Args:
        data_dir (str): Path to the folder where the data is located.
        api_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
        mlp_output_channels (int): Size of each output of the first linear layer.
        mashup_embed_channels (int): Size of each embedding vector of mashup.
        lr (float): Learning rate (default: :obj:`1e-3`).
        weight_decay (float): weight decay (default: :obj:`1e-5`).
    """
    def __init__(
        self,
        data_dir,
        api_embed_path: str,
        mlp_output_channels: int,
        mashup_embed_channels: int,
        lr: float ,
        weight_decay: float = 1e-5,
        gnn_layers = 10,
        api_graph=True,  
        api_out_channels =128,
        hidden_channels = 512,
        mashup_first_embed_channels = 768,
        mashup_new_embed_channels = 128,
        api_new_embed_channels = 128,
        heads = 1, 
        edge_message_agg='mean',
        edge_message = True,
        num_candidates = 932  
    ):
        r"""A MLP with two linear layers.

        Args:
            data_dir (str): Path to the folder where the data is located.
            api_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
            mlp_output_channels (int): Size of each output of the first linear layer.
            mashup_embed_channels (int): Size of each embedding vector of mashup
            lr: Learning rate (default: :obj:`1e-3`).
            weight_decay: weight decay (default: :obj:`1e-5`).
        """
        super(MLP_NEW, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.register_buffer('api_embeds', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.num_api = self.api_embeds.size(0)

        self.api_out_channels = api_out_channels

        self.gnn_layers = gnn_layers     
        self.heads = heads
        self.mashup_first_embed_channels = mashup_first_embed_channels
        self.hidden_channels = hidden_channels
        self.edge_message = edge_message
        self.edge_message_agg = edge_message_agg
        self.mashup_new_embed_channels = mashup_new_embed_channels
        self.api_new_embed_channels = api_new_embed_channels
        self.mlp_output_channels = mlp_output_channels
        self.num_candidates = num_candidates

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = torchmetrics.F1Score(top_k=5)

        self.mashup_path = ""

        self.register_buffer('H', torch.zeros(self.num_api, self.api_out_channels))
        zeros(self.H) 
        
        self.file_path = ''
        self.n = 0
        self._build_layers()



    def _build_layers(self):
        self.align_mlp = nn.Sequential(
            nn.Linear(self.mashup_first_embed_channels, self.hidden_channels),#128
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.api_out_channels)#512
        )
        self.api_reduce_dem_mlp = nn.Sequential(
            nn.Linear(self.api_embeds.size(1), self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.api_out_channels)
        )
        self.Hmlp = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        gnn_layers = []
        for i in range(self.gnn_layers):
            gnn_layers.append(TransformerConv(self.api_out_channels,
                                              int(self.api_out_channels / self.heads), self.heads,
                                              edge_dim=self.api_out_channels if self.edge_message else None))
            gnn_layers.append(LayerNorm(self.api_out_channels))
        self.gnns = nn.ModuleList(gnn_layers)
       
        ##############################################################
        hid_chan = 512
        hid_chan2 = 384
        hid_chan3 = 256
        hid_chan4 = 128
        hid_chan5 = 64
        hid_chan6 = 32
        self.linear = nn.Sequential(
            ResidualBlock(self.mashup_new_embed_channels + self.api_new_embed_channels, hid_chan, hid_chan2),
            ResidualBlock(hid_chan2, hid_chan3, hid_chan4),
            ResidualBlock(hid_chan4, hid_chan5, hid_chan6),
            nn.Linear(hid_chan6, 1)
        )
        
        ##############################################################
        metric = CosineSimilarity()
        processors = [KNNSparsify(450), NonLinearize(non_linearity='relu'), Symmetrize(), Normalize(mode='sym')]
        activation = nn.ReLU()
        self.gsl_learner = AttLearner(metric, processors, 10, 768, activation)

        processors2 = [KNNSparsify(40), NonLinearize(non_linearity='relu'), Symmetrize(), Normalize(mode='sym')]
        self.weight = 0.6
        self.mashup_learner = AttLearner(metric, processors2, 10, 768, activation) 
        #######################################################################################################################################################
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def training_step(self, batch: Any) -> STEP_OUTPUT:
        mashups_idx, labels = batch
        batch_size = mashups_idx.size(0)

        mashups = self.mashup_learning(mashups_idx)
        api_ori = self.api_reduce_dem_mlp(self.api_embeds) #  [768] -> [128]    [932,128]

        #######################################
        edge_index, edge_attr = self.format_samples(mashups, labels) 

        self.edge = edge_index

        adj = self.gsl_learner(self.api_embeds)
        edge_index1 , edge_attr1 , empt= self.mix_bing(adj, edge_index, edge_attr)
        if empt == 1:
            edge_index = edge_index1
            edge_attr = edge_attr1
        ###########################################################

        edge_index = edge_index.cuda()
        edge_attr = edge_attr.cuda()
        
        for i in range(self.gnn_layers):
            apis = self.gnns[i * 2](api_ori, edge_index, edge_attr)
            apis = self.gnns[i * 2 + 1](apis)

        ########################################

        Hid = torch.cat((apis, self.H), dim=-1).requires_grad_()
        Hid = torch.cat((Hid, api_ori), dim=-1).requires_grad_()
        Hid = self.Hmlp(Hid) #[932,256] - [932,128]

        self.H = Hid.detach()
        apis = Hid

        mashups_fin = mashups.unsqueeze(1).repeat(1, self.num_api, 1)#[batch,932,128]
        apis_fin = apis.unsqueeze(0).repeat(batch_size, 1, 1)#[batch,932,128]
        input_feature = torch.cat((mashups_fin, apis_fin), dim=-1).requires_grad_()#[batch,932,256]

        input_feature = input_feature.view(batch_size * self.num_candidates, 256)

        preds = self.linear(input_feature).requires_grad_()

        preds = preds.view(batch_size, self.num_candidates, -1)

        preds = preds.view(batch_size, self.num_api)
        loss = self.criterion(preds, labels.float())
        
        if torch.isnan(loss):
            with open(self.file_path, 'a') as file:
                file.write(f' {loss}\n')
                file.write(f' {preds}\n')
                file.write(f' {labels}\n')
        
        self.log('train/loss', loss)

        return loss

    
    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups_idx, labels = batch
        batch_size = mashups_idx.size(0)

        mashups = self.mashup_learning(mashups_idx)
        
        Hid = self.H
        apis = Hid

        mashups_fin = mashups.unsqueeze(1).repeat(1, self.num_api, 1)#[batch,932,128]
        apis_fin = apis.unsqueeze(0).repeat(batch_size, 1, 1)#[batch,932,128]
        input_feature = torch.cat((mashups_fin, apis_fin), dim=-1).requires_grad_()#[batch,932,256]4

        input_feature = input_feature.view(batch_size * self.num_candidates, 256)

        preds = self.linear(input_feature).requires_grad_()

        preds = preds.view(batch_size, self.num_candidates, -1)

        preds = preds.view(batch_size, self.num_api)

        self.log('val/F1', self.f1(preds, labels), on_step=False, on_epoch=True, prog_bar=False)
        


    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups_idx, labels = batch
        batch_size = mashups_idx.size(0)

        mashups = self.mashup_learning(mashups_idx)
      
        Hid = self.H
        apis = Hid

        mashups_fin = mashups.unsqueeze(1).repeat(1, self.num_api, 1)#[batch,932,128]
        apis_fin = apis.unsqueeze(0).repeat(batch_size, 1, 1)#[batch,932,128]
        input_feature = torch.cat((mashups_fin, apis_fin), dim=-1).requires_grad_()#[batch,932,256]4

        input_feature = input_feature.view(batch_size * self.num_candidates, 256)

        preds = self.linear(input_feature).requires_grad_()

        preds = preds.view(batch_size, self.num_candidates, -1)

        preds = preds.view(batch_size, self.num_api)

        return {
            'preds': preds,
            'targets': labels
        }

    def configure_optimizers(self):
        optimizer_model = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer_model
        
    def update_H(self, edge_index, edge_attr):
        
        adj = self.gsl_learner(self.api_embeds)
        edge_index , edge_attr , notempty = self.mix_bing(adj, edge_index, edge_attr)
        if notempty == 0:
            return 
        edge_index = edge_index.cuda()
        edge_attr = edge_attr.cuda()
        api_ori = self.api_reduce_dem_mlp(self.api_embeds)

        for i in range(self.gnn_layers):
            api = self.gnns[i * 2](api_ori, edge_index, edge_attr )
            api = self.gnns[i * 2 + 1](api)
        
        Hid = torch.cat((api, self.H), dim=-1)
        Hid = torch.cat((Hid, api_ori), dim=-1)
        self.H = self.Hmlp(Hid)
        

    def format_samples(self, Xs, Ys, **kwargs):
        """
        Xs: [B, C]  # C: message channel
        Ys: [B, N]  # N: API numbers
        """
        edge_message = {}
        for x, y in zip(Xs, Ys):
            invoked_apis = y.nonzero(as_tuple=True)[0]  
            invoked_apis = invoked_apis.cpu().numpy()
            for api_i in invoked_apis:
                for api_j in invoked_apis:
                    edge_message[(api_i, api_j)] = edge_message.get((api_i, api_j), []) + [x]
        src, dst, edge_attr = [], [], []
        for (u, v), messages in edge_message.items():
            
            src.append(u)
            dst.append(v)
            if self.edge_message_agg == 'mean':
                edge_attr.append(torch.mean(torch.stack(messages, dim=0), dim=0))
            else:
                edge_attr.append(torch.sum(torch.stack(messages, dim=0), dim=0))
        src = torch.tensor(src)
        dsc = torch.tensor(dst)
       
        edge_index = torch.stack([src, dsc], dim=0)
        edge_index = edge_index.to(self.device)
        edge_attr = torch.stack(edge_attr, dim=0).type_as(Xs)
        return edge_index, edge_attr          


    def mix_bing(self, adj, edge_index, edge_attr):
        notempty = 0
        filtered_edge_index = [[], []]
        filtered_edge_attr = []
        for i in range(edge_index.size(1)):
            src = edge_index[0][i]
            dst = edge_index[1][i]
            if adj[src, dst] != 0:
                continue
         
                filtered_edge_index[0].append(src)
                filtered_edge_index[1].append(dst)
                filtered_edge_attr.append(edge_attr[i] * 0.0001)


            else :
                notempty = 1
                filtered_edge_index[0].append(src)
                filtered_edge_index[1].append(dst)
                filtered_edge_attr.append(edge_attr[i])
            
        if notempty == 0 :
            return torch.tensor(filtered_edge_index), filtered_edge_attr, notempty
        filtered_edge_index = torch.tensor(filtered_edge_index)
        filtered_edge_attr = torch.stack(filtered_edge_attr)
        return filtered_edge_index, filtered_edge_attr, notempty

    def mashup_learning(self, mashup_idx):
        
        mashups = np.load(os.path.join(self.mashup_path)) 
        mashups = torch.from_numpy(mashups).cuda()
        adj_matrix = self.mashup_learner(mashups)

        eye = torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        adj_matrix = adj_matrix * (1 - eye)
        row_sums = adj_matrix.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1
        normalized_adj_matrix = adj_matrix / row_sums * (1 - self.weight)
        
        mashups = self.weight * mashups + torch.matmul(normalized_adj_matrix, mashups)

        mashups = self.align_mlp(mashups[mashup_idx])
        return mashups
    
