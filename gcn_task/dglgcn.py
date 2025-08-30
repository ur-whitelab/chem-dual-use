from dataclasses import dataclass
from functools import partial
import json
import os
import random
import urllib.request

import exmol
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops, rdmolfiles
from scipy.stats import spearmanr
import selfies as sf
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
#from dgllife.utils import SMILESToBigraph, smiles_to_bigraph, mol_to_bigraph # it's broken as of Feb 13 2023
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer


os.environ['DGLBACKEND'] = 'pytorch'
os.environ['OMP_NUM_THREADS'] = '1' # to ensure reproducible results and that DGL doesn't use OpenMP & introduce randomness 
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

urllib.request.urlretrieve(
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "./lipophilicity.csv",
)

# set random seeds for reproducibility
def set_seeds(seed):
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.Generator().manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

@dataclass
class Config:
    lr: float = 0.005
    hdim: int = 128
    split: float = 0.1 # 10/10/80 test val train
    # batch_size
    epochs: int = 60
    patience: int = 5
    min_delta: float = 1e-4 # for early stopping
    loss_func: str = 'mse' # 'mse' or 'rmse'

"""
functions copied from dgllife github: 
    - mol_to_graph, 
    - construct_bigraph_from_mol, 
    - mol_to_bigraph 
    
    (using them directly from them doesn't work for some reason)

"""
def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer,
                 canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0):
    if mol is None:
        print('Invalid mol found')
        return None

    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))

    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    if num_virtual_nodes > 0:
        num_real_nodes = g.num_nodes()
        real_nodes = list(range(num_real_nodes))
        g.add_nodes(num_virtual_nodes)

        # Change Topology
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)
        g.add_edges(virtual_src, virtual_dst)

        for nk, nv in g.ndata.items():
            nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
            nv[-num_virtual_nodes:, -1] = 1
            g.ndata[nk] = nv

        for ek, ev in g.edata.items():
            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev

    return g

def construct_bigraph_from_mol(mol, add_self_loop=False):
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    if add_self_loop:
        nodes = g.nodes().tolist()
        src_list.extend(nodes)
        dst_list.extend(nodes)

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=True,
                   explicit_hydrogens=False,
                   num_virtual_nodes=0):
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

def largest_mol(smiles):
  # remove ions from SMILES by getting the largest molecule part
  ss = smiles.split('.')
  ss.sort(key = lambda a: len(a))
  return ss[-1]


def get_noised_SMILES(smi, min_score, max_score, num_samples, preset='medium', ignore_fail_error=True, max_tries=10):
    # get a different SMILES with similarity score between min_score and max_score
    # first come first serve
    if min_score == 1.0 and max_score == 1.0:
        return smi, 1.0, True
    method_kwargs = {
        "num_samples": num_samples,
        "min_mutations": 1,
        "max_mutations": 1,
    }
    num_tries = 0
    while num_tries < max_tries:
        examples = exmol.sample_space(
            smi,
            preset=preset,
            f=lambda x: 0,
            batched=False,
            quiet=True,
            method_kwargs=method_kwargs
        )
        smiles = [e.smiles for e in examples[1:]] # ignore first with 1.0 score
        scores = [e.similarity for e in examples[1:]]
        for smile, score in zip(smiles, scores):
            if min_score < score < max_score:
                return smile, score, True
        if max_score <= 0.3:
            # increase mutations for higher chance with low similarities
            method_kwargs['max_mutations'] += 1
        method_kwargs['num_samples'] += 10
        num_tries += 1

    msg = f'Cannot find another SMILES between {min_score} and {max_score} for this SMILES!'
    msg += f'\nSMILES: {smi}'
    if ignore_fail_error == False:
        raise Exception(msg)
    return smi, 1.0, False

class LipophilicityDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='lipodata')

    def process(self):
        '''
        Extract smiles & labels from data in .csv file.
        DGL graph is created for each smiles string.
        '''
        lipodata = pd.read_csv("./lipophilicity.csv")
        smiles = lipodata.smiles
        labels = lipodata.exp
        self.graphs = []
        self.labels = []
        for smi, label in zip(smiles, labels):
            smi = largest_mol(smi)
            mol = Chem.MolFromSmiles(smi)
            g = mol_to_bigraph(mol,  
                               node_featurizer=CanonicalAtomFeaturizer('feat'),
                               edge_featurizer=CanonicalBondFeaturizer('feat'),
                               explicit_hydrogens=False, #note: loss worsened with this being True
                               add_self_loop=False,
                               )
            if g != None:
                self.graphs.append(g)
                self.labels.append(label)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        graph_path = os.path.join(self.save_path,'dgldata.bin' )
        dgl.save_graphs(graph_path,self.graphs,{'labels':torch.FloatTensor(self.labels)})

    def load(self):
        graph_path = os.path.join(self.save_path,'dgldata.bin' )
        self.graphs, label_dict = dgl.load_graphs(graph_path)
        self.labels = label_dict['labels']
        
    def has_cache(self):
        graph_path = os.path.join(self.save_path,'dgldata.bin' )
        return os.path.exists(graph_path)

class dgldataset(DGLDataset):
    # no noise, convert smiles and labels into DGL dataset
    def __init__(self, data):
        self.data = data
        super().__init__(name='somedata')

    def process(self):
        '''
        DGL graph is created for each smiles string.
        '''
        self.graphs = []
        self.labels = []

        for _, row in self.data.iterrows():
            smi = row['smiles']
            label = row['labels']
            smi = largest_mol(smi)
            mol = Chem.MolFromSmiles(smi)
            g = mol_to_bigraph(mol,  
                               node_featurizer=CanonicalAtomFeaturizer('feat'),
                               edge_featurizer=CanonicalBondFeaturizer('feat'),
                               explicit_hydrogens=False, # loss reduces when it's False
                               add_self_loop=False,
                               )
            if g is not None:
                self.graphs.append(g)
                self.labels.append(label)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)

class Xnoised_dataset(DGLDataset):
    def __init__(self, data, scorerange, threshold, targetregion,preset='medium'):
        # dataset must be in form of tuple (smiles,labels)
        #self.noiselevel = noiselevel
        self.data = data
        self.scorerange = scorerange
        self.threshold = threshold
        self.targetregion = targetregion
        self.preset = preset
        super().__init__(name='xnoised_data',raw_dir = '.', verbose=True)
    
    def process(self):
        minscore, maxscore = self.scorerange
        self.graphs = []
        self.labels = []
        self.smiles = []
        self.rawsmiles = []
        self.similarityscores = []
        noised_smiles_count = 0
        fail_count = 0
        for _, row in self.data.iterrows():
            smi = row['smiles']
            label = row['labels']
            smi = largest_mol(smi)
            self.rawsmiles.append(smi)

            score = None
            if minscore == 1.0 and maxscore == 1.0:
                score = 1.0
            # replace SMILES with noisy SMILES if it's in the target region
            elif self.targetregion == 'below':
                if label < self.threshold:
                    noised_smiles, score, success = get_noised_SMILES(
                        smi, minscore, maxscore, num_samples=15, preset=self.preset
                    )
                    if success == False: 
                        # sometimes it fails to get desired score - drop that data point
                        fail_count += 1
                        continue
                    smi = noised_smiles
                    noised_smiles_count += 1

            elif self.targetregion == 'above':
                if label > self.threshold:
                    noised_smiles, score, success = get_noised_SMILES(
                        smi, minscore, maxscore, num_samples=15, preset=self.preset
                    )
                    if success == False:
                        fail_count += 1
                        continue
                    smi = noised_smiles
                    noised_smiles_count += 1
            else: 
                raise Exception("'targetregion' can only take 'below' or 'above'")
            
            #make a SMILES graph
            mol = Chem.MolFromSmiles(smi)
            g = mol_to_bigraph(mol,  
                               node_featurizer=CanonicalAtomFeaturizer('feat'),
                               edge_featurizer=CanonicalBondFeaturizer('feat'),
                               add_self_loop=False,
                               )
            self.smiles.append(smi)
            self.graphs.append(g)
            self.labels.append(label)
            self.similarityscores.append(score)
        if noised_smiles_count > 0:
            print(f'Noised {noised_smiles_count} smiles out of data with {len(self.smiles)} smiles.')
            print(f'While adding feature noise, {fail_count} data points failed to '
                'generate SMILES within desired similarity scores. Removed from dataset.')

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)

class Ynoised_dataset(DGLDataset):
    def __init__(self, data, mag_noise=1.0, threshold=0, targetregion='above'):
        self.data = data
        self.mag_noise = mag_noise
        self.threshold = threshold
        self.targetregion = targetregion
        super().__init__(name='ynoised_data')

    def process(self):
        self.graphs = []
        self.smiles = []
        self.labels = []
        self.rawlabels = [] # non-noised labels

        for _, row in self.data.iterrows():
            smi = row['smiles']
            label = row['labels']
            self.rawlabels.append(label)

            if self.targetregion == 'below':
                if label < self.threshold:
                    label = self.mag_noise * np.random.normal() + label
            elif self.targetregion == 'above':
                if label > self.threshold:
                    label = self.mag_noise * np.random.normal() + label
            else: 
                raise Exception("'targetregion' argument can only accept 'below' or 'above'")
            
            #make a SMILES graph
            smi = largest_mol(smi)
            mol = Chem.MolFromSmiles(smi)
            g = mol_to_bigraph(mol,  
                               node_featurizer=CanonicalAtomFeaturizer('feat'),
                               edge_featurizer=CanonicalBondFeaturizer('feat'),
                               add_self_loop=False,
                               )
            self.smiles.append(smi)
            self.graphs.append(g)
            self.labels.append(label)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.dense = nn.Linear(h_feats, 1) # reduce to dim 1
    
    def forward(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g,h))
        g.ndata['h'] = h
        h = dgl.mean_nodes(g,'h') # readout by averaging
        h = self.dense(h)
        return h.squeeze()

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, verbose=True):
        # look at validation loss to check whether it isn't improving for few steps
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if verbose:
                    print("Early stopping. No improvement in validation loss.")
                return True
        return False

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def compute_threshold_from_split(labels, split, region):
    """
    Computes a threshold from a given split ratio based on labels and 
    desired sensitive region.
    """
    sorted_labels = np.sort(labels)
    if region == 'above': 
        index = int(len(sorted_labels) * (1-split)) # top split as sensitive
    elif region == 'below':
        index = int(len(sorted_labels) * split) # bottom split as sensitive
    else:
        raise ValueError("Invalid region. Must be 'above' or 'below'.")
    threshold = sorted_labels[index]
    return threshold

def train(model, model_config, train_data, val_data, verbose = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = GraphDataLoader(train_data, batch_size=5, drop_last=False)
    val_dataloader = GraphDataLoader(val_data, batch_size=5, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    early_stopper = EarlyStopper(patience=model_config.patience, min_delta=model_config.min_delta)
    if model_config.loss_func == 'mse':
        calc_loss = nn.MSELoss()
        use_rmse = False
    elif model_config.loss_func == 'rmse':
        calc_loss = RMSELoss()
        use_rmse = True
    if verbose:
        print(f'Using {model_config.loss_func} as loss function.')

    best_val_loss = float('inf')
    train_loss = []
    val_loss = []
    for epoch in range(model_config.epochs):
        curr_train_loss = 0.0
        for batched_data in train_dataloader:
            model.train()
            graphs, labels = batched_data
            graphs = graphs.to(device)
            # forward
            pred = model(graphs,graphs.ndata['feat'].float())
            loss = calc_loss(pred, labels.float())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curr_train_loss += loss.item()
        curr_train_loss /= len(train_dataloader)
        train_loss.append(curr_train_loss)

        curr_val_loss = 0.0
        for batched_data in val_dataloader:
            _, loss = evaluate(model, batched_data, use_rmse=use_rmse)
            curr_val_loss += loss
        curr_val_loss /= len(val_dataloader)
        val_loss.append(curr_val_loss)

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            #torch.save(model.state_dict(), 'best_model.pt')
        if epoch % 5 == 0 and verbose:
            print(f'Epoch: {epoch:02d}, Train Loss: {curr_train_loss:.4f},' 
                  f'Val Loss: {curr_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
        if early_stopper.early_stop(curr_val_loss, verbose):
            break
    return train_loss, val_loss

def local_loss(y,yhat, ymin, ymax, use_rmse=True):
    y, yhat = torch.tensor(y), torch.tensor(yhat)
    mask = (y >= ymin) & (y <= ymax)
    local_y = y[mask]
    local_yhat = yhat[mask]
    local_mse = F.mse_loss(local_yhat, local_y)
    if use_rmse:
        local_rmse = torch.sqrt(local_mse)
        return local_rmse.item()
    return local_mse.item()

def local_spearman(y, yhat, threshold, above=True):
    # Filter y and yhat based on the threshold
    if above:
        mask = np.array(y) > threshold
    else:
        mask = np.array(y) <= threshold

    local_y = np.array(y)[mask]
    local_yhat = np.array(yhat)[mask]

    if len(local_y) > 1:  # Ensure there are at least 2 data points
        corr, _ = spearmanr(local_y, local_yhat)
        return corr
    else:
        return np.nan  # Not enough data points for a valid correlation

def evaluate(model, data, use_rmse=True):
    '''
    calculate loss on validation or test data

    Args:
    - model: the trained model
    - data: the validation or test data
    - use_rmse: whether to use RMSE or MSE as loss function. True is RMSE, False is MSE
    '''
    model.eval()
    graphs, labels = data
    if isinstance(graphs, list):
        # if data is a list rather than batched data
        preds = []
        for graph in graphs:
            with torch.no_grad():
                pred = model(graph, graph.ndata['feat'].float())
            preds.append(pred.item())
        loss = F.mse_loss(torch.FloatTensor(preds), torch.FloatTensor(labels))   
    else:
        # if data is batched data or single data point
        with torch.no_grad():
            preds = model(graphs, graphs.ndata['feat'].float())
            labels = labels.float()
            if labels.shape != preds.shape:
                if preds.dim() == 0:
                    preds = torch.tensor([preds])
                else:
                    print('Shapes of labels and predictions do not match.')
                    print("preds shape: ", preds.shape)
                    print("labels shape: ", labels.shape)
                    print("preds: ", preds)
                    print("labels: ", labels)
        loss = F.mse_loss(preds, labels.float())
    if use_rmse:
        loss = torch.sqrt(loss)
    return preds, loss.item() 

def omit_sensitive_data_rank_based(df, threshold, omit_region, omit_frac=1):
    # Omits a fraction of sensitive data based on ranking in a specified region.
    # First omits the most sensitive data points (highest or lowest)
    if omit_frac == 0:
        return df
    if omit_region not in ['above', 'below']:
        raise ValueError("omit_region must be either 'above' or 'below'")
    if not 0 <= omit_frac <= 1:
        raise ValueError("omit_fraction must be in the range [0,1]")
    
    if omit_region == 'above':
        is_sensitive = df['labels'] > threshold
    else:
        is_sensitive = df['labels'] < threshold

    sensitive_data = df[is_sensitive]
    non_sensitive_data = df[~is_sensitive]
    if omit_frac == 1:
        return non_sensitive_data
    
    # rank sensitive data based on labels and calculate the number of items to omit
    if omit_region == 'above':
        # for 'above', higher labels are more sensitive, so sort descending
        sensitive_data_sorted = sensitive_data.sort_values(by='labels', ascending=False)
    else:
        # for 'below', lower labels are more sensitive, so sort ascending
        sensitive_data_sorted = sensitive_data.sort_values(by='labels', ascending=True)

    num_to_omit = int(len(sensitive_data_sorted) * omit_frac)
    kept_sensitive_data = sensitive_data_sorted[num_to_omit:]
    result_df = pd.concat([kept_sensitive_data, non_sensitive_data], ignore_index=True)

    return result_df


def omit_sensitive_data(df, threshold, omit_region, omit_frac=1):
    if omit_frac == 0:
        return df
    if omit_region not in ['above', 'below']:
        raise ValueError("omit_region must be either 'above' or 'below'")
    if not 0 <= omit_frac <= 1:
        raise ValueError("omit_fraction must be in the range [0,1]")

    if omit_region == 'above':
        sensitive_data = df[df['labels'] > threshold]
        non_sensitive_data = df[df['labels'] <= threshold]
    else:
        sensitive_data = df[df['labels'] < threshold]
        non_sensitive_data = df[df['labels'] >= threshold]
    
    if omit_frac == 1:
        return non_sensitive_data
    else:
        omit_size = int(omit_frac * len(sensitive_data))
        kept_indices = random.sample(list(sensitive_data.index), len(sensitive_data) - omit_size)
        kept_sensitive_data = sensitive_data.loc[kept_indices]
    print(f"Kept {len(kept_sensitive_data)} out of {len(sensitive_data)} sensitive data points.")
    omitted_fraction = (1 - len(kept_sensitive_data) / max(len(sensitive_data), 1)) * 100
    print(f"Fraction omitted: {omitted_fraction:.2f}%")

    result_df = pd.concat([kept_sensitive_data, non_sensitive_data], ignore_index=True)
    return result_df

# we don't use this - too random to study its effects - great for censoring purpose though
def omit_sensitive_data_probabilistic(df, threshold, omit_region, omit_frac=1):
    if omit_frac == 0:
        return df
    if omit_region not in ['above', 'below']:
        raise ValueError("omit_region must be either 'above' or 'below'")
    if not 0 <= omit_frac <= 1:
        raise ValueError("omit fraction must be in the range [0,1]")

    # separate data into sensitive and non-sensitive based on threshold and region
    if omit_region == 'above':
        is_sensitive = df['labels'] > threshold
    else:
        is_sensitive = df['labels'] < threshold
    
    sensitive_data = df[is_sensitive]
    non_sensitive_data = df[~is_sensitive]

    if omit_frac == 1:
        return non_sensitive_data

    # probabilistically keep or omit sensitive data points, aka. probabilistic omission
    keep_mask = np.random.rand(len(sensitive_data)) > omit_frac
    kept_sensitive_data = sensitive_data[keep_mask]

    print(f"Kept {len(kept_sensitive_data)} out of {len(sensitive_data)} sensitive data points.")
    omitted_fraction = (1 - len(kept_sensitive_data) / max(len(sensitive_data), 1)) * 100
    print(f"Fraction omitted: {omitted_fraction:.2f}%")

    result_df = pd.concat([kept_sensitive_data, non_sensitive_data], ignore_index=True)
    return result_df

# no noise for control
def no_noise_train_wrapper(
        censor_region, 
        censor_threshold=None,
        censor_split=None,
        model_config=None,
        jobname='zero_noise',
        dir_name=None,
        random_state=None, 
        verbose=True,
        rawdata = None,
        separate_train_path=None,  # Path to pre-split training data
        separate_val_path=None,    # Path to pre-split validation data
        separate_test_path=None,
        save_split_data = True,
    ):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if model_config == None:
        model_config = Config()

    if separate_train_path and separate_val_path and separate_test_path:
        train_subset = pd.read_csv(separate_train_path)
        val_subset = pd.read_csv(separate_val_path)
        test_subset = pd.read_csv(separate_test_path)
        if censor_threshold is not None:
            print(f'Censor threshold = {censor_threshold}')
    else:
        if rawdata is None:
            raise ValueError("rawdata must be provided if pre-split data paths are not specified.")
        # compute threshold from sensitive/non-sensitive split
        if censor_threshold is None and censor_split is not None:
            censor_threshold = compute_threshold_from_split(
                rawdata.labels, censor_split, censor_region
            )
            print(f"With {censor_split:.0%} censored split, ")
            print(f"censor threshold = {censor_threshold}")

        # split data
        train_subset, nontrain_subset = train_test_split(rawdata, test_size=2*model_config.split, random_state=random_state)
        val_subset, test_subset = train_test_split(nontrain_subset, test_size=0.5, random_state=random_state)
        if dir_name and save_split_data:
            os.makedirs(f"{dir_name}/data", exist_ok=True)
            train_subset.to_csv(f"{dir_name}/data/train_data_{random_state}.csv", index=False)
            val_subset.to_csv(f"{dir_name}/data/val_data_{random_state}.csv", index=False)
            test_subset.to_csv(f"{dir_name}/data/test_data_{random_state}.csv", index=False)

    train_data = dgldataset(train_subset)
    val_data = dgldataset(val_subset)
    test_data = dgldataset(test_subset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1,_ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], model_config.hdim).to(device)

    train_loss, val_loss = train(model, model_config, train_data, val_data, verbose)

    # evaluate the test data
    ytest = test_data.labels
    ymin = min(ytest)
    ymax = max(ytest)
    yhat, rmse = evaluate(model, (test_data.graphs, ytest))
    corr = spearmanr(ytest,yhat)[0]

    if censor_threshold is not None:
        lower_rmse = local_loss(ytest,yhat, ymin, censor_threshold)
        upper_rmse = local_loss(ytest,yhat, censor_threshold, ymax) 
        lower_corr = local_spearman(ytest, yhat, censor_threshold, above=False)
        upper_corr = local_spearman(ytest, yhat, censor_threshold, above=True)
    
    # dump results
    if dir_name:
        print(f'Saving results to {dir_name} folder')
        train_curve_path = f'{dir_name}/trainingcurve_{jobname}.json'
        parity_plot_path = f'{dir_name}/parityplotdata_{jobname}.json'
    else:
        print('Saving results to current directory')
        train_curve_path = f'trainingcurve_{jobname}.json'
        parity_plot_path = f'parityplotdata_{jobname}.json'
    epochs = np.arange(model_config.epochs).tolist()
    with open(train_curve_path,'w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)

    with open(parity_plot_path,'w') as f2:
        if censor_threshold:
            json.dump([str(rmse),lower_rmse,upper_rmse, corr, lower_corr, upper_corr, ytest,yhat],f2)
        else:
            json.dump([str(rmse), corr, ytest,yhat],f2)
    if censor_threshold:
        return rmse, lower_rmse, upper_rmse, corr, lower_corr, upper_corr
    return rmse, corr

# omission method baseline
def omit_train_wrapper(
        censor_region, 
        censor_split=None,
        censor_threshold=None,
        omit_fraction=1, 
        model_config=None, 
        jobname=None,
        dir_name="omit_results",
        random_state=None,
        verbose=True,
        rawdata = None, 
        separate_train_path=None,  # Path to pre-split training data
        separate_val_path=None,    # Path to pre-split validation data
        separate_test_path=None,
        save_split_data = True,
    ):
    os.makedirs(dir_name, exist_ok=True)
    if model_config == None:
        model_config = Config()
    if jobname == None:
        jobname = 'omitted_data'

    if separate_train_path and separate_val_path and separate_test_path:
        train_subset = pd.read_csv(separate_train_path)
        val_subset = pd.read_csv(separate_val_path)
        test_subset = pd.read_csv(separate_test_path)
        if censor_threshold is None:
            raise ValueError("censor_threshold is required if pre-split data paths are provided.")
        print(f'Censor threshold = {censor_threshold}')
    else:
        if rawdata is None:
            raise ValueError("rawdata must be provided if pre-split data paths are not specified.")
        # compute threshold from sensitive/non-sensitive split
        if censor_threshold is None and censor_split is not None:
            censor_threshold = compute_threshold_from_split(
                rawdata.labels, censor_split, censor_region
            )
            print(f"With {censor_split:.0%} censored split, ")
            print(f"censor threshold = {censor_threshold}")

        # split data
        train_subset, nontrain_subset = train_test_split(rawdata, test_size=2*model_config.split, random_state=random_state)
        val_subset, test_subset = train_test_split(nontrain_subset, test_size=0.5, random_state=random_state)
        if save_split_data:
            os.makedirs(f"{dir_name}/data", exist_ok=True)
            train_subset.to_csv(f"{dir_name}/data/train_data_{random_state}.csv", index=False)
            val_subset.to_csv(f"{dir_name}/data/val_data_{random_state}.csv", index=False)
            test_subset.to_csv(f"{dir_name}/data/test_data_{random_state}.csv", index=False)

    # omit data in sensitive region 
    print(f'Omitting {omit_fraction:.0%} of sensitive data in training data...')
    filtered_train_subset = omit_sensitive_data(
        train_subset, censor_threshold, censor_region, omit_fraction
    )
    print(f'Omitting {omit_fraction:.0%} of sensitive data in validation data...')
    filtered_val_subset = omit_sensitive_data(
        val_subset, censor_threshold, censor_region, omit_fraction
    )
    
    with open(f'{dir_name}/omittedlabels.json','w') as f0:
        original_labels = train_subset['labels'].tolist()
        filtered_labels = filtered_train_subset['labels'].tolist()
        json.dump([original_labels, filtered_labels], f0)

    # convert to DGL dataset
    test_data = dgldataset(test_subset)
    val_data = dgldataset(filtered_val_subset)
    train_data = dgldataset(filtered_train_subset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1,_ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], model_config.hdim).to(device)

    train_loss, val_loss = train(model, model_config, train_data, val_data, verbose)

    # evaluate the test data
    ytest = test_data.labels
    ymin = min(ytest)
    ymax = max(ytest)
    yhat, rmse = evaluate(model, (test_data.graphs, ytest))
    lower_rmse = local_loss(ytest,yhat, ymin, censor_threshold) # region below threshold
    upper_rmse = local_loss(ytest,yhat, censor_threshold, ymax) # region above threshold 
    corr = spearmanr(ytest,yhat)[0]
    lower_corr = local_spearman(ytest, yhat, censor_threshold, above=False)
    upper_corr = local_spearman(ytest, yhat, censor_threshold, above=True)

    epochs = np.arange(model_config.epochs).tolist()
    with open(f'{dir_name}/trainingcurve_{jobname}.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)

    with open(f'{dir_name}/parityplotdata_{jobname}.json','w') as f2:
        json.dump([str(rmse),lower_rmse,upper_rmse,corr,lower_corr,upper_corr,ytest,yhat],f2)

    return rmse, lower_rmse, upper_rmse, corr, lower_corr, upper_corr

def xnoise_train_wrapper( 
        simscore_range, 
        censor_region, 
        censor_split=None,
        censor_threshold=None, 
        model_config=None,
        jobname=None, 
        dir_name="xnoise_results",
        random_state=None,
        verbose=True,
        rawdata = None, 
        separate_train_path=None,  # Path to pre-split training data
        separate_val_path=None,    # Path to pre-split validation data
        separate_test_path=None,
        save_split_data = True,
    ):
    os.makedirs(dir_name, exist_ok=True)
    if model_config == None:
        model_config = Config()
    if jobname == None:
        jobname = f'similarityscore{str(simscore_range[0])}-{str(simscore_range[1])}'
    
    if separate_train_path and separate_val_path and separate_test_path:
        train_subset = pd.read_csv(separate_train_path)
        val_subset = pd.read_csv(separate_val_path)
        test_subset = pd.read_csv(separate_test_path)
        if censor_threshold is None:
            raise ValueError("censor_threshold is required if pre-split data paths are provided.")
        print(f'Censor threshold = {censor_threshold}')
    else:
        if rawdata is None:
            raise ValueError("rawdata must be provided if pre-split data paths are not specified.")
        # compute threshold from sensitive/non-sensitive split
        if censor_threshold is None and censor_split is not None:
            censor_threshold = compute_threshold_from_split(
                rawdata.labels, censor_split, censor_region
            )
            print(f"With {censor_split:.0%} censored split, ")
            print(f"censor threshold = {censor_threshold}")

        # split data
        train_subset, nontrain_subset = train_test_split(rawdata, test_size=2*model_config.split, random_state=random_state)
        val_subset, test_subset = train_test_split(nontrain_subset, test_size=0.5, random_state=random_state)
        if save_split_data:
            os.makedirs(f"{dir_name}/data", exist_ok=True)
            train_subset.to_csv(f"{dir_name}/data/train_data_{random_state}.csv", index=False)
            val_subset.to_csv(f"{dir_name}/data/val_data_{random_state}.csv", index=False)
            test_subset.to_csv(f"{dir_name}/data/test_data_{random_state}.csv", index=False)

    train_data = Xnoised_dataset(
        train_subset,
        scorerange=simscore_range, 
        threshold=censor_threshold, 
        targetregion=censor_region, 
        preset='medium'
    )
    val_data = Xnoised_dataset(
        val_subset,
        scorerange=simscore_range, 
        threshold=censor_threshold, 
        targetregion=censor_region, 
        preset='medium'
    )
    test_data = dgldataset(test_subset) # remain unmodified for evaluation purpose

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1,_ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], model_config.hdim).to(device)

    # train
    train_loss, val_loss = train(model, model_config, train_data, val_data, verbose)

    # evaluate raw test data
    ytest = test_data.labels
    ymin = min(ytest)
    ymax = max(ytest)
    yhat, rmse = evaluate(model, (test_data.graphs, ytest))
    lower_rmse = local_loss(ytest,yhat, ymin, censor_threshold) 
    upper_rmse = local_loss(ytest,yhat, censor_threshold, ymax)  
    corr = spearmanr(ytest,yhat)[0]
    lower_corr = local_spearman(ytest, yhat, censor_threshold, above=False)
    upper_corr = local_spearman(ytest, yhat, censor_threshold, above=True)
    
    # bonus: evaluate how it predicts 'fake' test data (data with censored sensitive info)
    noised_test_data = Xnoised_dataset(
        test_subset,
        scorerange=simscore_range, 
        threshold=censor_threshold, 
        targetregion=censor_region, 
        preset='medium'
    )
    noised_ytest = noised_test_data.labels
    ymin = min(noised_ytest)
    ymax = max(noised_ytest)
    noised_yhat, noised_rmse = evaluate(model, (noised_test_data.graphs, noised_ytest))
    noised_lower_rmse = local_loss(noised_ytest,noised_yhat, ymin, censor_threshold)
    noised_upper_rmse = local_loss(noised_ytest,noised_yhat, censor_threshold, ymax)

    # save data in json files
#     with open(f'{dir_name}/noisedsmiles_{jobname}.json','w') as f0:
#         json.dump([
#             noised_test_data.labels,
#             noised_test_data.similarityscores, 
#             #noised_test_data.iterations # outdated, not using iterations anymore
#         ],f0)
    
    epochs = np.arange(model_config.epochs).tolist()
    with open(f'{dir_name}/trainingcurve_{jobname}.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)

    with open(f'{dir_name}/parityplotdata_{jobname}.json','w') as f2: 
        json.dump([str(rmse),lower_rmse, upper_rmse,corr,lower_corr,upper_corr,ytest,yhat],f2)

    with open(f'{dir_name}/parityplotdata_noisedtestdata_{jobname}.json','w') as f3: 
        # note that this uses NOISED test data
        json.dump([
            str(noised_rmse), 
            noised_lower_rmse, 
            noised_upper_rmse, 
            noised_ytest, 
            noised_yhat
        ],f3)
    print('Results are stored in json files.')
    return rmse, lower_rmse, upper_rmse, corr, lower_corr, upper_corr

def ynoise_train_wrapper(
        noise_level, 
        censor_region, 
        censor_split=None,
        censor_threshold=None, 
        model_config=None,
        jobname=None,
        dir_name="ynoise_results",
        random_state=None, 
        verbose=True,
        rawdata = None,
        separate_train_path=None,  # Path to pre-split training data
        separate_val_path=None,    # Path to pre-split validation data
        separate_test_path=None,
        save_split_data = True,
    ):
    os.makedirs(dir_name, exist_ok=True)
    if model_config == None:
        model_config = Config()
    if jobname == None:
        jobname = f'ynoise{noise_level}'

    if separate_train_path and separate_val_path and separate_test_path:
        train_subset = pd.read_csv(separate_train_path)
        val_subset = pd.read_csv(separate_val_path)
        test_subset = pd.read_csv(separate_test_path)
        if censor_threshold is None:
            raise ValueError("censor_threshold is required if pre-split data paths are provided.")
        print(f'Censor threshold = {censor_threshold}')
    else:
        if rawdata is None:
            raise ValueError("rawdata must be provided if pre-split data paths are not specified.")
        # compute threshold from sensitive/non-sensitive split
        if censor_threshold is None and censor_split is not None:
            censor_threshold = compute_threshold_from_split(
                rawdata.labels, censor_split, censor_region
            )
            print(f"With {censor_split:.0%} censored split, ")
            print(f"censor threshold = {censor_threshold}")

        # split data
        train_subset, nontrain_subset = train_test_split(rawdata, test_size=2*model_config.split, random_state=random_state)
        val_subset, test_subset = train_test_split(nontrain_subset, test_size=0.5, random_state=random_state)
        if save_split_data:
            os.makedirs(f"{dir_name}/data", exist_ok=True)
            train_subset.to_csv(f"{dir_name}/data/train_data_{random_state}.csv", index=False)
            val_subset.to_csv(f"{dir_name}/data/val_data_{random_state}.csv", index=False)
            test_subset.to_csv(f"{dir_name}/data/test_data_{random_state}.csv", index=False)

    test_data = dgldataset(test_subset)
    
    # noising the training and val data
    train_data = Ynoised_dataset(
        train_subset,
        mag_noise=noise_level, 
        threshold=censor_threshold, 
        targetregion=censor_region
    )
    val_data = Ynoised_dataset(
        val_subset,
        mag_noise=noise_level, 
        threshold=censor_threshold, 
        targetregion=censor_region
    )

    # save raw and noised labels for sanity check
    raw_y = train_data.rawlabels
    noised_y = train_data.labels
    with open(f'{dir_name}/noisedlabels_traindata_{jobname}.json','w') as f0:
        json.dump([raw_y,noised_y],f0)

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1,_ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], model_config.hdim).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # train
    train_loss, val_loss = train(model, model_config, train_data, val_data, verbose)

    # evaluate raw test data
    ytest = test_data.labels
    ymin = min(ytest)
    ymax = max(ytest)
    yhat, rmse = evaluate(model, (test_data.graphs, ytest))
    lower_rmse = local_loss(ytest,yhat, ymin, censor_threshold)
    upper_rmse = local_loss(ytest,yhat, censor_threshold, ymax) 
    corr = spearmanr(ytest,yhat)[0]
    lower_corr = local_spearman(ytest, yhat, censor_threshold, above=False)
    upper_corr = local_spearman(ytest, yhat, censor_threshold, above=True)

    # bonus: to see how it predicts noised data, whether it 'censors' the noised region
    noised_test_data = Ynoised_dataset(
        test_subset,
        mag_noise=noise_level, 
        threshold=censor_threshold, 
        targetregion=censor_region
    )
    noised_ytest = noised_test_data.labels
    ymin = min(noised_ytest)
    ymax = max(noised_ytest)
    noised_yhat, noised_rmse = evaluate(model, (noised_test_data.graphs, noised_ytest))
    noised_lower_rmse = local_loss(noised_ytest,noised_yhat, ymin, censor_threshold) 
    noised_upper_rmse = local_loss(noised_ytest,noised_yhat, censor_threshold, ymax)

    # save data in json files
    epochs = np.arange(model_config.epochs).tolist()
    with open(f'{dir_name}/trainingcurve_{jobname}.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)
    
    with open(f'{dir_name}/parityplotdata_{jobname}.json','w') as f2: 
        json.dump([str(rmse),lower_rmse,upper_rmse,corr,lower_corr,upper_corr,ytest,yhat],f2)

    with open(f'{dir_name}/parityplotdata_noisedtestdata_{jobname}.json','w') as f3: 
        # note that this is with NOISED test data
        json.dump([
            str(noised_rmse), 
            noised_lower_rmse, 
            noised_upper_rmse, 
            noised_ytest, 
            noised_yhat
        ],f3)
    print('json files are loaded!')

    return rmse, lower_rmse, upper_rmse, corr, lower_corr, upper_corr

# if __name__ == "__main__":
#     lipodata = pd.read_csv("./lipophilicity.csv")
#     rawdata = list(zip(lipodata.smiles,lipodata.exp))
#     xnoise_train_wrapper(rawdata, num_epochs=10, simscore_range=[0.6,0.8], threshold=2, targetregion='above', jobname='0')




            
