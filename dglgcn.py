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

# hyperparameter (some)
hdim = 128
lr = 0.005

# set random seeds for reproducibility
def set_seeds(seed):
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.Generator().manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


# functions copied from dgllife github: mol_to_graph, construct_bigraph_from_mol, mol_to_bigraph (using them directly from them doesn't work for some reason)
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

class Xnoised_dataset(DGLDataset):
    def __init__(self, data, scorerange=[0.5,0.75], threshold=0, targetregion='above',preset='medium'):
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
        for smi, label in self.data:
            score = 'None'
            bestscore = 0
            iter = None
            smi = largest_mol(smi)
            self.rawsmiles.append(smi)

            if self.targetregion == 'below':
                if label < self.threshold:
                    noised_smiles = None
                    iter = 0
                    while noised_smiles == None:
                        examples = exmol.sample_space(smi, preset=self.preset,  f=lambda x: 0, batched=False, quiet=True, num_samples=15)
                        examples.pop(0) # first has 1.0 similiarity
                        smiles = [e.smiles for e in examples]
                        scores = [e.similarity for e in examples]
                        for i in range(len(scores)):
                            if minscore < scores[i] < maxscore:
                                noised_smiles = smiles[i]
                                score = scores[i]
                            elif max(scores) > bestscore: # need to fix so SMILES within the score range is selected randomly
                                bestscore = max(scores)
                                idx = scores.index(bestscore)
                                max_smi = smiles[idx]
                        iter += 1
                        if iter > 10:
                            score = bestscore
                            noised_smiles = max_smi
                            break
                    smi = noised_smiles

            elif self.targetregion == 'above':
                if label > self.threshold:
                    noised_smiles = None
                    iter = 0
                    while noised_smiles == None:
                        examples = exmol.sample_space(smi, preset=self.preset,  f=lambda x: 0, batched=False, quiet=True, num_samples=15)
                        examples.pop(0) # first has 1.0 similiarity
                        smiles = [e.smiles for e in examples]
                        scores = [e.similarity for e in examples]
                        for i in range(len(scores)):
                            if minscore < scores[i] < maxscore:
                                noised_smiles = smiles[i]
                                score = scores[i]
                            elif max(scores) > bestscore:
                                bestscore = max(scores)
                                idx = scores.index(bestscore)
                                max_smi = smiles[idx]
                        iter += 1
                        if iter > 10:
                            score = bestscore
                            noised_smiles = max_smi
                            break
                    smi = noised_smiles

            else: 
                raise Exception("'targetregion' argument can only accept 'below' or 'above'")
            
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

def train(model, data, optimizer,device):
    model.train()
    graphs, labels = data
    graphs = graphs.to(device)
    # forward
    pred = model(graphs,graphs.ndata['feat'].float())
    loss = F.mse_loss(pred, labels.float())
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def local_mse(y,yhat, min, max):
    y, yhat = torch.tensor(y), torch.tensor(yhat)
    mask = (y >= min) & (y <= max)
    #local_indices = np.where(mask)[0]
    local_y = y[mask] #[local_indices]
    local_yhat = yhat[mask]
    local_mse = F.mse_loss(local_yhat, local_y)
    return local_mse.item()

def evaluate(model, data):
    model.eval()
    graphs, labels = data
    with torch.no_grad():
      pred = model(graphs, graphs.ndata['feat'].float())
    loss = F.mse_loss(pred, labels.float())
    return pred, loss.item() 

def filterdata_by_label(dataset, threshold, omitregion):
    if omitregion not in ['above', 'below']:
        raise ValueError("omitregion must be either 'above' or 'below'")
    elif omitregion == 'above':
        return [(smiles,label) for (smiles, label) in dataset if label < threshold]
    else:
        return [(smiles,label) for (smiles, label) in dataset if label > threshold]

def baseline_nonoise(rawdata, num_epochs=50, threshold=2, random_state=None):
    train_subset, val_subset, test_subset = split_dataset(rawdata, frac_list=[0.8, 0.1, 0.1], shuffle=True,random_state=random_state)
    train_data = dgldataset(train_subset)
    val_data = dgldataset(val_subset)
    test_data = dgldataset(test_subset)

    train_dataloader = GraphDataLoader(train_data, batch_size=5, drop_last=False)
    val_dataloader = GraphDataLoader(val_data, batch_size=5, drop_last=False)
    test_dataloader = GraphDataLoader(test_data, batch_size=5, drop_last=False)

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1,_ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], hdim).to(device)
    #model = biggerGCN(graph1.ndata['feat'].shape[1], hdim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    best_val_loss = float('inf')
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        curr_train_loss = 0.0
        for batched_data in train_dataloader:
            curr_train_loss += train(model, batched_data, optimizer, device)
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
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:02d}, Train Loss: {curr_train_loss:.4f}, Val Loss: {curr_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
    
    # evaluate the test data
    yhat = []
    for graph in test_data.graphs:
        model.eval()
        pred = model(graph, graph.ndata['feat'].float())
        yhat.append(pred.item())
    ytest = test_data.labels
    mse = F.mse_loss(torch.FloatTensor(yhat), torch.FloatTensor(ytest)).item()
    upper_mse = local_mse(ytest,yhat, threshold, 100) #mse for region above threshold 
    lower_mse = local_mse(ytest,yhat, -100, threshold) #mse for region below threshold

 
    epochs = np.arange(num_epochs).tolist()
    with open(f'trainingcurve_no_noise.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)

    with open(f'parityplotdata_no_noise.json','w') as f2:
        json.dump([str(mse),lower_mse,upper_mse,ytest,yhat],f2)

    return mse, lower_mse, upper_mse


# omission method baseline
def baseline_filtereddata(rawdata, num_epochs=50, threshold=2, targetregion='above', random_state=None):
    train_subset, val_subset, test_subset = split_dataset(rawdata, frac_list=[0.8, 0.1, 0.1], shuffle=True,random_state=random_state)
    filtered_train_subset = filterdata_by_label(train_subset, threshold=threshold, omitregion=targetregion)
    filtered_val_subset = filterdata_by_label(val_subset, threshold=threshold, omitregion=targetregion)
    
    _, raw_y = zip(*train_subset)
    _, filtered_y = zip(*filtered_train_subset)
    with open(f'omittedlabels.json','w') as f0:
        json.dump([raw_y,filtered_y],f0)

    # convert to DGL dataset
    test_data = dgldataset(test_subset)
    val_data = dgldataset(filtered_val_subset)
    train_data = dgldataset(filtered_train_subset)

    train_dataloader = GraphDataLoader(train_data, batch_size=5, drop_last=False)
    val_dataloader = GraphDataLoader(val_data, batch_size=5, drop_last=False)
    test_dataloader = GraphDataLoader(test_data, batch_size=5, drop_last=False)

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1,_ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], hdim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    best_val_loss = float('inf')
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        curr_train_loss = 0.0
        for batched_data in train_dataloader:
            curr_train_loss += train(model, batched_data, optimizer, device)
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
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:02d}, Train Loss: {curr_train_loss:.4f}, Val Loss: {curr_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
    
    # evaluate the test data
    yhat = []
    for graph in test_data.graphs:
        model.eval()
        pred = model(graph, graph.ndata['feat'].float())
        yhat.append(pred.item())
    ytest = test_data.labels
    mse = F.mse_loss(torch.FloatTensor(yhat), torch.FloatTensor(ytest)).item()
    upper_mse = local_mse(ytest,yhat, threshold, 100) #mse for region above threshold 
    lower_mse = local_mse(ytest,yhat, -100, threshold) #mse for region below threshold

 
    epochs = np.arange(num_epochs).tolist()
    with open(f'trainingcurve_filterdata.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)

    with open(f'parityplotdata_filterdata.json','w') as f2:
        json.dump([str(mse),lower_mse,upper_mse,ytest,yhat],f2)

    return mse, lower_mse, upper_mse



def xnoise_train_wrapper(rawdata, num_epochs=50, simscore_range=[0.6,0.8], threshold=2, targetregion='above', jobname='0', random_state=None):
    # split data and create DGLdataset
    train_subset, val_subset, test_subset = split_dataset(rawdata, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=random_state)
    test_rawdata = dgldataset(test_subset)

    train_data = Xnoised_dataset(train_subset,scorerange=simscore_range, threshold=threshold, targetregion=targetregion, preset='medium')
    val_data = Xnoised_dataset(val_subset,scorerange=simscore_range, threshold=threshold, targetregion=targetregion, preset='medium')
    test_data = Xnoised_dataset(test_subset,scorerange=simscore_range, threshold=threshold, targetregion=targetregion, preset='medium') # optional, raw test data is still preserved
    
    train_dataloader = GraphDataLoader(train_data, batch_size=5, drop_last=False)
    val_dataloader = GraphDataLoader(val_data, batch_size=5, drop_last=False)
    test_dataloader = GraphDataLoader(test_data, batch_size=5, drop_last=False)

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1,_ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], hdim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # train
    best_val_loss = float('inf')
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        curr_train_loss = 0.0
        for batched_data in train_dataloader:
            curr_train_loss += train(model, batched_data, optimizer, device)
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
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:02d}, Train Loss: {curr_train_loss:.4f}, Val Loss: {curr_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
    
    # to see how it predicts noised data, whether it 'censors' the noised region
    yhat = []
    for graph in test_data.graphs:
        model.eval()
        pred = model(graph, graph.ndata['feat'].float())
        yhat.append(pred.item())
    ytest = test_data.labels
    mse = F.mse_loss(torch.FloatTensor(yhat), torch.FloatTensor(ytest)).item()
    upper_mse = local_mse(ytest,yhat, threshold, 100) #mse for region above threshold 
    lower_mse = local_mse(ytest,yhat, -100, threshold) #mse for region below threshold
    
    # evaluate the test data
    rawyhat = []
    for graph in test_rawdata.graphs:
        model.eval()
        pred = model(graph, graph.ndata['feat'].float())
        rawyhat.append(pred.item())
    rawytest = test_rawdata.labels
    mse_raw = F.mse_loss(torch.FloatTensor(rawyhat), torch.FloatTensor(rawytest)).item()
    raw_upper_mse = local_mse(rawytest,rawyhat, threshold, 100) #mse for region above threshold 
    raw_lower_mse = local_mse(rawytest,rawyhat, -100, threshold) #mse for region below threshold

    # save data in json files
    with open(f'noisedsmiles_{jobname}.json','w') as f0:
        json.dump([test_data.labels,test_data.similarityscores, test_data.iterations],f0)
    
    epochs = np.arange(num_epochs).tolist()
    with open(f'trainingcurve_{jobname}.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)

    with open(f'parityplotdata_{jobname}.json','w') as f2: # note that this is with NOISED test data
        json.dump([str(mse),lower_mse, upper_mse,ytest,yhat],f2)

    with open(f'parityplotdata_raw_{jobname}.json','w') as f3:
        json.dump([str(mse_raw), raw_lower_mse, raw_upper_mse, rawytest, rawyhat],f3)
    print('json files are loaded!')

    return mse_raw, raw_lower_mse, raw_upper_mse

def ynoise_train_wrapper(rawdata, num_epochs=50, mag_noise=1.5, threshold=2, targetregion='above', random_state=None):
    # split data
    train_subset, val_subset, test_subset = split_dataset(rawdata, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=random_state)
    test_rawdata = dgldataset(test_subset)

    # noising the training and val data
    train_data = Ynoised_dataset(train_subset,mag_noise=mag_noise, threshold=threshold, targetregion=targetregion)
    val_data = Ynoised_dataset(val_subset,mag_noise=mag_noise, threshold=threshold, targetregion=targetregion)
    test_data = Ynoised_dataset(test_subset,mag_noise=mag_noise, threshold=threshold, targetregion=targetregion) # optional, raw test data is still preserved

    raw_y = train_data.rawlabels
    noised_y = train_data.labels
    with open(f'noisedlabels_{mag_noise}.json','w') as f0:
        json.dump([raw_y,noised_y],f0)

    # create mini batches
    train_dataloader = GraphDataLoader(train_data, batch_size=5, drop_last=False)
    val_dataloader = GraphDataLoader(val_data, batch_size=5, drop_last=False)
    test_dataloader = GraphDataLoader(test_data, batch_size=5, drop_last=False) 

    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1,_ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], hdim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # train
    best_val_loss = float('inf')
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        curr_train_loss = 0.0
        for batched_data in train_dataloader:
            curr_train_loss += train(model, batched_data, optimizer, device)
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
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:02d}, Train Loss: {curr_train_loss:.4f}, Val Loss: {curr_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
    
    # to see how it predicts noised data, whether it 'censors' the noised region
    ytest=[]
    yhat=[]
    mse = 0.0
    for batched_data in test_dataloader:
        batched_graphs, labels = batched_data
        ytest.extend(labels.detach().tolist())
        pred, loss = evaluate(model, batched_data)
        yhat.extend(pred.detach().tolist())
        mse += loss
    mse = mse / len(test_dataloader)
    upper_mse = local_mse(ytest,yhat, threshold, 100) #mse for region above threshold 
    lower_mse = local_mse(ytest,yhat, -100, threshold) #mse for region below threshold

    # evaluate the test data
    rawyhat = []
    for graph in test_rawdata.graphs:
        model.eval()
        pred = model(graph, graph.ndata['feat'].float())
        rawyhat.append(pred.item())
    rawytest = test_rawdata.labels
    mse_raw = F.mse_loss(torch.FloatTensor(rawyhat), torch.FloatTensor(rawytest)).item()
    raw_upper_mse = local_mse(rawytest,rawyhat, threshold, 100) #mse for region above threshold 
    raw_lower_mse = local_mse(rawytest,rawyhat, -100, threshold) #mse for region below threshold

    # save data in json files
    epochs = np.arange(num_epochs).tolist()
    with open(f'trainingcurve_mag{mag_noise}.json','w') as f1:
        json.dump([epochs,train_loss,val_loss],f1)
    
    with open(f'parityplotdata_mag{mag_noise}.json','w') as f2: # note that this is with NOISED test data
        json.dump([str(mse),lower_mse,upper_mse,ytest,yhat],f2)

    with open(f'parityplotdata_raw_mag{mag_noise}.json','w') as f3:
        json.dump([str(mse_raw), raw_lower_mse, raw_upper_mse, rawytest, rawyhat],f3)

    return mse_raw, raw_lower_mse, raw_upper_mse

if __name__ == "__main__":
    lipodata = pd.read_csv("./lipophilicity.csv")
    rawdata = list(zip(lipodata.smiles,lipodata.exp))
    xnoise_train_wrapper(rawdata, num_epochs=10, simscore_range=[0.6,0.8], threshold=2, targetregion='above', jobname='0')




            
