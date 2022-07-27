import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import rgb2hex
import seaborn as sns

from pipeline_functions import training_data_select
from pipeline_functions import viz_training_data
from pipeline_functions import one_hot_encode
from pipeline_functions import cell_type_classifier
from pipeline_functions import process_label

# reading in object -> converted from RDS (sceasy), preprocessed already & embedded in UMAP space
file_dir = '/Users/katebridges/PycharmProjects/test/ttx.h5ad'
new_bos = sc.read(file_dir)

# fig params
sc.set_figure_params(figsize=(5, 5), fontsize=20)

# VIZ by experimental condition
cmap = matplotlib.cm.get_cmap('viridis')
my_pal0 = {'YR': cmap(0.25),
           'YR_TTx': cmap(0.75)}
sc.pl.umap(new_bos, color='Sample', palette=my_pal0)

# SET UP selection of training/validation set for neural network
marker = ['Cd3d', 'Cd3e', 'Cd3g', 'Cd4', 'Foxp3', 'Il2ra', 'Cd8a',
          'Ncr1', 'Klri2',
          'Blnk',
          'Csf1r', 'Sirpa',
          'Fscn1', 'Cacnb3',
          'Cxcr2', 'Lcn2',
          'Dcn', 'Prrx1', 'Ptprc',
          'Ccnd1', 'Lgals7', 'Cyb5r3',
          'Klrb1c'
          ]

cell_types = ['CD4+ T cell', 'CD8+ T cell', 'Treg', 'NK cell', 'B cell', 'Macrophage', 'DC', 'Neutrophil', 'Fibroblast',
              'Tumor cell']

celltypes = np.zeros((10, len(marker)))
celltypes[0, :7] = [1, 1, 1, 1, -1, -1, -1]  # CD4+
celltypes[1, :7] = [1, 1, 1, -1, -1, -1, 1]  # CD8+
celltypes[2, :7] = [1, 1, 1, 1, 1, 0, -1]  # Treg
celltypes[3, :3] = [-1, -1, -1]  # NK
celltypes[3, 7:9] = [1, 1]
celltypes[3, 22] = 1
# lymphoid lineage cells need to be off for Csf1r, Sirpa
celltypes[:4, 10:12] = -1*np.ones((4, 2))
celltypes[4, 9] = 1  # B cell (small population)
celltypes[5, 10:12] = [1, 1]  # macrophage
celltypes[6, 12:14] = [1, 1]  # DC
celltypes[7, 14:16] = [1, 1]  # neut
celltypes[8, 16:19] = [0, 1, -1]  # cancer-associated fibroblast
celltypes[9, 18:22] = [-1, 1, 1, 1]  # tumor cell

tot_lab, tot_ideal_ind, tot_traindata, tot_testdata = training_data_select(new_bos, marker, celltypes, cell_types,
                                                                           np.arange(10))
# visualizing training/validation data
cmap_ = sns.color_palette('Spectral', 10)
viz_training_data(new_bos, tot_lab, tot_ideal_ind, cell_types, new_bos.obsm['X_umap'], 'UMAP', cmap_,
                  'Training/validation sets (~20%)', (6, 5), 0.75)

# FEEDFORWARD NEURAL NETWORK FOR CELL TYPE ANNOTATION, VISUALIZATION
learning_rate = 0.025  # altering learning rate to change how much neural net can adjust during each training epoch
training_epochs = 500
batch_size = 100
display_step = 5

# split into testing/training
tot_lab_onehot = one_hot_encode(tot_lab)
all_train_ind = np.array([])
ideal_ = np.argmax(tot_lab_onehot, axis=1)
train_split = 0.5
for k in np.unique(ideal_):
    all_ind = np.where(ideal_ == k)[0]  # randomly select half for training, other half goes to validation
    train_ind = np.random.choice(all_ind, round(train_split*len(all_ind)), replace=False)
    all_train_ind = np.concatenate((all_train_ind, train_ind))

total_predicted_lab, tot_prob, colorm, pred = cell_type_classifier(tot_lab_onehot, tot_traindata,
                                                                   tot_testdata,
                                                                   all_train_ind,
                                                                   learning_rate, training_epochs, batch_size,
                                                                   display_step)

# reordering cell type labels and filtering by probability
total_lab, total_prob = process_label(tot_prob, tot_lab, total_predicted_lab, tot_ideal_ind, new_bos, 0.8)

# write results as metadata
cluster2annotation = {0: 'CD4+ T cell',
                      1: 'CD8+ T cell',
                      2: 'Treg',
                      3: 'NK cell',
                      4: 'B cell',
                      5: 'Macrophage',
                      6: 'DC',
                      7: 'Neutrophil',
                      8: 'Fibroblast',
                      9: 'Tumor cell',
                      -1: 'Poorly classified'
                      }
new_bos.obs['nn_80'] = pd.Series(total_lab.astype('int')).map(cluster2annotation).values

# write to file
new_bos.write('/Users/katebridges/PycharmProjects/test/ttx_annotated20210211.h5ad')

# VIZ by assigned cell type
cmap_ = sns.color_palette('Spectral', 10)
celltype_dict = {cell_types[p]: cmap_[p] for p in np.arange(len(cell_types))}
sc.pl.umap(new_bos, color='nn_80', palette=celltype_dict)

# DOTPLOT to demonstrate marker expr
marker_genes_dict = {'Immune cell': ['Ptprc'],
                     'General T cell': ['Cd3d', 'Cd3e', 'Cd3g'],
                     'CD4+ T cell': ['Cd4'],
                     'CD8+ T cell': ['Cd8a'],
                     'Treg': ['Foxp3', 'Il2ra'],
                     'NK cell': ['Ncr1', 'Klri2', 'Klrb1c'],
                     'B cell': ['Blnk'],
                     'Macrophage': ['Csf1r', 'Sirpa'],
                     'DC': ['Fscn1', 'Cacnb3'],
                     'Neutrophil': ['Cxcr2', 'Lcn2'],
                     'Fibroblast': ['Dcn', 'Prrx1'],
                     'Tumor cell': ['Ccnd1', 'Lgals7', 'Cyb5r3']
                     }

# reordering cell types for visualization purposes
ordered_types = cell_types = ['CD4+ T cell', 'CD8+ T cell', 'Treg', 'NK cell', 'B cell', 'Macrophage',
                              'DC', 'Neutrophil', 'Fibroblast', 'Tumor cell', 'Poorly classified']
new_bos.obs['nn_ordered'] = pd.Categorical(new_bos.obs['nn_80'], categories=ordered_types, ordered=True)

matplotlib.rcParams.update({'font.size': 15})
sc.pl.dotplot(new_bos, marker_genes_dict, 'nn_ordered', dendrogram=False, log=True, var_group_rotation=70)

# VIZ expression of cognate receptors for therapy across assigned cell types
ttx_receptors = ['Pdcd1', 'Csf1r', 'Cd40']
sc.pl.stacked_violin(new_bos, ttx_receptors, groupby='nn_80', dendrogram=True, fig_size=(15, 15), swap_axes=True,
                     cmap='Reds', vmin=-1, vmax=2)

# SCORING AND VIZ by C/C activity (from serum data)
cc_geneset = ['Csf2', 'Ccl21b', 'Cxcl2', 'Csf1', 'Cx3cl1', 'Ifng', 'Cxcl1', 'Il12b', 'Csf3', 'Il6', 'Ccl5', 'Ccl11', 'Ccl2',
              'Ccl4', 'Il11', 'Ccl19', 'Tnf', 'Ccl22', 'Cxcl9', 'Ccl17', 'Timp1', 'Ccl20', 'Il16', 'Ccl12',
              'Lif', 'Il10', 'Cxcl10']
sc.tl.score_genes(new_bos, cc_geneset, score_name='CC_score', use_raw=True)
sc.pl.umap(new_bos, color='CC_score', cmap='seismic', vmin=-4, vmax=4)


