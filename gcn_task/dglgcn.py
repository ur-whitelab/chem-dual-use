import urllib.request
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset
#from dgllife.utils import SMILESToBigraph, smiles_to_bigraph, mol_to_bigraph # it's broken as of Feb 13 2023
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
from dgl.nn import GraphConv
from rdkit import Chem
from functools import partial
from rdkit.Chem import rdmolops, rdmolfiles
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import Subset, split_dataset
from dataclasses import dataclass
import os 
import json
import exmol
import random
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

# functions copied from dgllife github: 
#       mol_to_graph, construct_bigraph_from_mol, mol_to_bigraph 
# (using them directly from them doesn't work for some reason)
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

def get_noised_SMILES(smi, min_score, max_score, num_samples, preset='medium'):
    noised_smiles = None
    iter = 0
    while noised_smiles == None:
        examples = exmol.sample_space(
            smi, 
            preset=preset,  
            f=lambda x: 0, 
            batched=False, 
            quiet=True, 
            min_mutations=3,
            max_mutations=3,
            num_samples=num_samples,
        )
        examples.pop(0) # first has 1.0 similiarity
        smiles = [e.smiles for e in examples]
        scores = [e.similarity for e in examples]
        for i in range(len(scores)):
            if min_score < scores[i] < max_score:
                # first SMILES within the score range is selected
                noised_smiles = smiles[i] 
                score = scores[i]
            elif min(scores) > bestscore: 
                # if failed to get desired similiarity, 
                # save the lower similarity score closest to minimum desired score 
                bestscore = max(scores)
                idx = scores.index(bestscore)
                max_smi = smiles[idx]
        iter += 1
        if iter > 10:
            print(f'Cannot find SMILES between {min_score} and {max_score}!')
            print(f'Proceed with SMILES with next closest similarity score: {max_smi}')
            score = bestscore
            noised_smiles = max_smi
            break
    return noised_smiles, score

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
        Extract smiles & labels from data in .csv file.
        DGL graph is created for each smiles string.
        '''
        #smiles, labels = zip(*self.data)
        self.graphs = []
        self.labels = []
        for smi, label in self.data:
        #for smi, label in zip(smiles, labels):
            smi = largest_mol(smi)
            mol = Chem.MolFromSmiles(smi)
            g = mol_to_bigraph(mol,  
                               node_featurizer=CanonicalAtomFeaturizer('feat'),
                               edge_featurizer=CanonicalBondFeaturizer('feat'),
                               explicit_hydrogens=False, # loss reduces when it's False
                               add_self_loop=False,
                               )
            if g != None:
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
        self.iterations = []
        noised_smiles_count = 0
        fail_count = 0
        for smi, label in self.data:
            score = 'None'
            bestscore = 0
            iter = None
            smi = largest_mol(smi)
            self.rawsmiles.append(smi)

            # replace SMILES with noisy SMILES if it's in the target region
            if self.targetregion == 'below':
                if label < self.threshold:
                    noised_smiles, score = get_noised_SMILES(
                        smi, minscore, maxscore, num_samples=15, preset=self.preset
                    )
                    if score < minscore or score > maxscore:
                        fail_count += 1
                    smi = noised_smiles
                    noised_smiles_count += 1

            elif self.targetregion == 'above':
                if label > self.threshold:
                    noised_smiles, score = get_noised_SMILES(
                        smi, minscore, maxscore, num_samples=15, preset=self.preset
                    )
                    
                    if score < minscore or score > maxscore:
                        fail_count += 1
                    smi = noised_smiles
                    noised_smiles_count += 1
            else: 
                raise Exception("'targetregion' can only take 'below' or 'above'")
            
            #make a graph
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
            self.iterations.append(iter)
        print(f'Noised {noised_smiles_count} SMILES out of {len(self.smiles)} total')
        print(f'Out of {noised_smiles_count} noisy SMILES, {fail_count}' 
              'failures to generate SMILES within desired similarity scores.' 
              f'Similarity scores lower than {minscore} are used instead.')

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

        for smi, label in self.data:
            self.rawlabels.append(label)
            if self.targetregion == 'below':
                if label < self.threshold:
                    label = self.mag_noise * np.random.normal() + label
            elif self.targetregion == 'above':
                if label > self.threshold:
                    label = self.mag_noise * np.random.normal() + label
            else: 
                raise Exception("'targetregion' argument can only accept 'below' or 'above'")
            
            #make a graph
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

# def old_train(model, batched_data, optimizer,device):
#     model.train()
#     graphs, labels = batched_data
#     graphs = graphs.to(device)
#     # forward
#     pred = model(graphs,graphs.ndata['feat'].float())
#     loss = F.mse_loss(pred, labels.float())
#     # backward
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss.item()

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
    rmse = RMSELoss()

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
            loss = rmse(pred, labels.float())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curr_train_loss += loss.item()
        curr_train_loss /= len(train_dataloader)
        train_loss.append(curr_train_loss)

        curr_val_loss = 0.0
        for batched_data in val_dataloader:
            _, loss = evaluate(model, batched_data)
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

def local_rmse(y,yhat, ymin, ymax):
    rmse = RMSELoss()
    y, yhat = torch.tensor(y), torch.tensor(yhat)
    mask = (y >= ymin) & (y <= ymax)
    local_y = y[mask]
    local_yhat = yhat[mask]
    local_rmse = rmse(local_yhat, local_y)
    return local_rmse.item()

def evaluate(model, data):
    rmse = RMSELoss()
    model.eval()
    graphs, labels = data
    if isinstance(graphs, list):
        # if data is a list rather than batched data
        preds = []
        for graph in graphs:
            with torch.no_grad():
                pred = model(graph, graph.ndata['feat'].float())
            preds.append(pred.item())
        loss = rmse(torch.FloatTensor(preds), torch.FloatTensor(labels))   
    else:
        # if data is batched data or single data point
        with torch.no_grad():
            preds = model(graphs, graphs.ndata['feat'].float())
        loss = rmse(preds, labels.float())
    return preds, loss.item() 

# def filterdata_by_label(dataset, threshold, omitregion):
#     if omitregion not in ['above', 'below']:
#         raise ValueError("omitregion must be either 'above' or 'below'")
#     elif omitregion == 'above':
#         return [(smiles,label) for (smiles, label) in dataset if label < threshold]
#     else:
#         return [(smiles,label) for (smiles, label) in dataset if label > threshold]

def omit_sensitive_data(dataset, threshold, omit_region, omit_frac=1):
    if omit_region not in ['above', 'below']:
        raise ValueError("omit_region must be either 'above' or 'below'")
    if not 0 <= omit_frac <= 1:
        raise ValueError("omit_fraction must be in the range [0,1]")

    if omit_region == 'above':
        sensitive_data = [
            (smiles,label) for (smiles, label) in dataset if label > threshold
        ]
        non_sensitive_data = [
            (smiles,label) for (smiles, label) in dataset if label <= threshold
        ]
    else:
        sensitive_data = [
            (smiles,label) for (smiles, label) in dataset if label < threshold
        ]
        non_sensitive_data = [
            (smiles,label) for (smiles, label) in dataset if label >= threshold
        ]
    
    if omit_frac == 1:
        return non_sensitive_data
    else:
        omit_size = int(omit_frac * len(sensitive_data))
        kept_sensitive_data = random.sample(
            sensitive_data, len(sensitive_data) - omit_size
        )
    print(f"Kept {len(kept_sensitive_data)} sensitive data points out of " 
          f"{len(sensitive_data)} total (sensitive only).")
    print(f"Fraction omitted: "
          f"{(1 - len(kept_sensitive_data) / len(sensitive_data))*100}")
    return kept_sensitive_data + non_sensitive_data

def omit_sensitive_data_probabilistic(
        dataset, threshold, omit_region, omit_frac=1):
    # love the probabilistic omission method but it's too random to measure its effect
    if omit_region not in ['above', 'below']:
        raise ValueError("omit_region must be either 'above' or 'below'")
    if not 0 <= omit_frac <= 1:
        raise ValueError("omit fraction must be in the range [0,1]")

    # Separate data into sensitive and non-sensitive based on threshold and region
    if omit_region == 'above':
        sensitive_data = [
            (smiles, label) for (smiles, label) in dataset if label > threshold
        ]
        non_sensitive_data = [
            (smiles, label) for (smiles, label) in dataset if label <= threshold
        ]
    else:
        sensitive_data = [
            (smiles, label) for (smiles, label) in dataset if label < threshold
        ]
        non_sensitive_data = [
            (smiles, label) for (smiles, label) in dataset if label >= threshold
        ]
    if omit_frac == 1:
        return non_sensitive_data

    # Probabilistically keep or omit sensitive data points
    kept_sensitive_data = []
    for (smiles, label) in sensitive_data:
        if random.random() > omit_frac:
            kept_sensitive_data.append((smiles, label))
    print(f"Kept {len(kept_sensitive_data)} sensitive data points out of " 
          f"{len(sensitive_data)} total (sensitive only).")
    print(f"Fraction omitted: "
          f"{(1 - len(kept_sensitive_data) / len(sensitive_data))*100}")

    return kept_sensitive_data + non_sensitive_data

# no noise for control
def no_noise_train_wrapper(
        rawdata, 
        sensitive_region=None, 
        sensitive_threshold=None,
        sensitive_split=None,
        model_config=None,
        jobname='zero_noise',
        dir_name=None,
        random_state=None, 
        verbose=True,
    ):
    if model_config == None:
        model_config = Config()

    train_subset, val_subset, test_subset = split_dataset(
        rawdata, frac_list=[0.8, 0.1, 0.1], shuffle=True,random_state=random_state
    )
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

    # compute RMSE for sensitive/non-sensitive region if split or threshold is given
    if sensitive_split is not None:
        sensitive_threshold = compute_threshold_from_split(
            ytest, sensitive_split, sensitive_region
        )
        print(f"With {sensitive_split:.0%} sensitive split, "
              f"sensitive threshold = {sensitive_threshold}")
    if sensitive_threshold is not None:
        lower_rmse = local_rmse(ytest,yhat, ymin, sensitive_threshold)
        upper_rmse = local_rmse(ytest,yhat, sensitive_threshold, ymax) 
    
    # dump results
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
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
        if sensitive_threshold:
            json.dump([str(rmse),lower_rmse,upper_rmse,ytest,yhat],f2)
        else:
            json.dump([str(rmse),ytest,yhat],f2)

    return rmse, lower_rmse, upper_rmse

# omission method baseline
def omit_train_wrapper(
        rawdata, 
        censor_region, 
        censor_split=None,
        censor_threshold=None,
        omit_fraction=1, 
        model_config=None, 
        jobname=None,
        dir_name="omit_results",
        random_state=None,
        verbose=True,
    ):
    if model_config == None:
        model_config = Config()
    if jobname == None:
        jobname = 'omitted_data'

    # compute threshold from sensitive/non-sensitive split
    if censor_split is not None:
        labels = [label for _, label in rawdata]
        censor_threshold = compute_threshold_from_split(
            labels, censor_split, censor_region
        )
        print(f"With {censor_split:.0%} censored split, "
              f"censor threshold = {censor_threshold}")

    train_subset, val_subset, test_subset = split_dataset(
        rawdata, frac_list=[0.8, 0.1, 0.1], shuffle=True,random_state=random_state
    )

    # omit data in sensitive region 
    print(f'Omitting {omit_fraction:.0%} of sensitive data in training data...')
    filtered_train_subset = omit_sensitive_data(
        train_subset, censor_threshold, censor_region, omit_fraction
    )
    print(f'Omitting {omit_fraction:.0%} of sensitive data in validation data...')
    filtered_val_subset = omit_sensitive_data(
        val_subset, censor_threshold, censor_region, omit_fraction
    )
    
    _, raw_y = zip(*train_subset)
    _, filtered_y = zip(*filtered_train_subset)
    os.makedirs(dir_name, exist_ok=True)
    with open(f'{dir_name}/omittedlabels.json','w') as f0:
        json.dump([raw_y,filtered_y],f0)

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
    lower_rmse = local_rmse(ytest,yhat, ymin, censor_threshold) # region below threshold
    upper_rmse = local_rmse(ytest,yhat, censor_threshold, ymax) # region above threshold 

    epochs = np.arange(model_config.epochs).tolist()
    with open(f'{dir_name}/trainingcurve_{jobname}.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)

    with open(f'{dir_name}/parityplotdata_{jobname}.json','w') as f2:
        json.dump([str(rmse),lower_rmse,upper_rmse,ytest,yhat],f2)

    return rmse, lower_rmse, upper_rmse

def xnoise_train_wrapper(
        rawdata, 
        simscore_range, 
        censor_region, 
        censor_split=None,
        censor_threshold=None, 
        model_config=None,
        jobname=None, 
        dir_name="xnoise_results",
        random_state=None,
        verbose=True,
    ):
    if model_config == None:
        model_config = Config()
    if jobname == None:
        jobname = f'similarityscore{str(simscore_range[0])}-{str(simscore_range[1])}'

    # compute threshold from sensitive/non-sensitive split
    if censor_split is not None:
        labels = [label for _, label in rawdata]
        censor_threshold = compute_threshold_from_split(
            labels, censor_split, censor_region
        )
        print(f"With {censor_split:.0%} censored split, "
              f"censor threshold = {censor_threshold}")

    # split data
    train_subset, val_subset, test_subset = split_dataset(
        rawdata, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=random_state
    )
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
    lower_rmse = local_rmse(ytest,yhat, ymin, censor_threshold) 
    upper_rmse = local_rmse(ytest,yhat, censor_threshold, ymax)  
    
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
    noised_lower_rmse = local_rmse(noised_ytest,noised_yhat, ymin, censor_threshold)
    noised_upper_rmse = local_rmse(noised_ytest,noised_yhat, censor_threshold, ymax)

    # save data in json files
    os.makedirs(dir_name, exist_ok=True)
    with open(f'{dir_name}/noisedsmiles_{jobname}.json','w') as f0:
        json.dump([
            noised_test_data.labels,
            noised_test_data.similarityscores, 
            noised_test_data.iterations
        ],f0)
    
    epochs = np.arange(model_config.epochs).tolist()
    with open(f'{dir_name}/trainingcurve_{jobname}.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)

    with open(f'{dir_name}/parityplotdata_{jobname}.json','w') as f2: 
        json.dump([str(rmse),lower_rmse, upper_rmse,ytest,yhat],f2)

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
    return rmse, lower_rmse, upper_rmse

def ynoise_train_wrapper(
        rawdata,  
        noise_level, 
        censor_region, 
        censor_split=None,
        censor_threshold=None, 
        model_config=None,
        jobname=None,
        dir_name="ynoise_results",
        random_state=None, 
        verbose=True
    ):
    if model_config == None:
        model_config = Config()
    if jobname == None:
        jobname = f'ynoise{noise_level}'

    # compute threshold from sensitive/non-sensitive split
    if censor_split is not None:
        labels = [label for _, label in rawdata]
        censor_threshold = compute_threshold_from_split(
            labels, censor_split, censor_region
        )
        print(f"With {censor_split:.0%} censored split, "
              f"censor threshold = {censor_threshold}")

    # split data
    train_subset, val_subset, test_subset = split_dataset(
        rawdata, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=random_state
    )
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
    os.makedirs(dir_name, exist_ok=True)
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
    lower_rmse = local_rmse(ytest,yhat, ymin, censor_threshold)
    upper_rmse = local_rmse(ytest,yhat, censor_threshold, ymax) 

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
    noised_lower_rmse = local_rmse(noised_ytest,noised_yhat, ymin, censor_threshold) 
    noised_upper_rmse = local_rmse(noised_ytest,noised_yhat, censor_threshold, ymax)

    # save data in json files
    epochs = np.arange(model_config.epochs).tolist()
    with open(f'{dir_name}/trainingcurve_{jobname}.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)
    
    with open(f'{dir_name}/parityplotdata_{jobname}.json','w') as f2: 
        json.dump([str(rmse),lower_rmse,upper_rmse,ytest,yhat],f2)

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

    return rmse, lower_rmse, upper_rmse

# if __name__ == "__main__":
#     lipodata = pd.read_csv("./lipophilicity.csv")
#     rawdata = list(zip(lipodata.smiles,lipodata.exp))
#     xnoise_train_wrapper(rawdata, num_epochs=10, simscore_range=[0.6,0.8], threshold=2, targetregion='above', jobname='0')




            
