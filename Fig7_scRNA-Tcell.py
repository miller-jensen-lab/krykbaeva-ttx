import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import scirpy as ir
import anndata

# figure params
sc.set_figure_params(figsize=(5, 5), fontsize=15)

# reading in full dataset annotated by cell type
new_bos = sc.read('/Users/katebridges/PycharmProjects/test/ttx_annotated20210211.h5ad')

# reading in TCR sequencing data (with scirpy)
tcrs = []
samples = ['TWD1', 'TWD2']
for g in samples:
    tcr_sample = ir.io.read_10x_vdj('/Users/katebridges/Downloads/TTx_VDJ/{}_contig_annotations.csv'.format(g))
    new_obsnames = []
    for k in np.arange(len(tcr_sample.obs_names)):
        new_obsnames.append(tcr_sample.obs_names[k][:-1] + '{}'.format(g[3]))
    tcr_sample.obs_names = new_obsnames
    tcrs.append(tcr_sample)

# merge samples into one file
tcr_seq = anndata.concat(tcrs)
tcr_seq.obs_names_make_unique()

# merge with scRNA-seq data object
ir.pp.merge_with_ir(new_bos, tcr_seq)

# limit to T cells only moving forward here (only cells with TCRs)
t_ind = np.concatenate((np.where(new_bos.obs['nn_80'] == 'CD4+ T cell')[0],
                        np.where(new_bos.obs['nn_80'] == 'CD8+ T cell')[0],
                        np.where(new_bos.obs['nn_80'] == 'Treg')[0]
                        ))


def remove_recompute(adata):
    del adata.obsm['X_umap'], adata.obsm['X_diffmap'], adata.obsm['X_pca'], adata.obsp, adata.uns
    sc.tl.pca(adata, svd_solver='auto')
    sc.pp.neighbors(adata)  # using with default parameters
    sc.tl.umap(adata)
    return adata


t_cells = new_bos[t_ind, :]
# recomputing PCA, UMAP embeddings
t_cells = remove_recompute(t_cells)

# VIZ by expr condition, cell type
cmap = matplotlib.cm.get_cmap('viridis')
my_pal0 = {'YR': cmap(0.25),
           'YR_TTx': cmap(0.75)}

sc.pl.umap(t_cells, legend_loc='lower right', color='Sample', s=40, palette=my_pal0)

my_pal1 = {'CD4+ T cell': cmap(0.01),
           'CD8+ T cell': cmap(0.5),
           'Treg': cmap(0.99)}
sc.pl.umap(t_cells, legend_loc='lower right', color='nn_80', s=40, palette=my_pal1)

# STACKED VIOLIN of genes of interest
sample_celltype = []
for k in range(t_cells.shape[0]):
    sample_celltype.append(t_cells.obs['nn_80'][k] + ' ' + t_cells.obs['Sample'][k])

t_cells.obs['condID'] = pd.Categorical(sample_celltype)
tcell_expr = ['Ifng', 'Mki67', 'Prf1', 'Gzma', 'Gzmb', 'Tox']
sc.pl.stacked_violin(t_cells, tcell_expr, groupby='condID', dendrogram=True, fig_size=(15, 15), swap_axes=True, cmap='Reds', use_raw=False, vmin=-0.5, vmax=1)

# TCR ANALYSES - based on scirpy tutorial
ir_cmap = {'True': 'C1',
           'False': 'C0',
           'None': 'C2'
           }
sc.pl.umap(t_cells, color='has_ir', palette=ir_cmap)

# finding dual TCRs
ir.tl.chain_qc(t_cells)
ax = ir.pl.group_abundance(t_cells, groupby="receptor_subtype", target_col="Sample")
ax1 = ir.pl.group_abundance(t_cells, groupby="chain_pairing", target_col="Sample")

# defining clonotypes
ir.pp.ir_dist(t_cells)
ir.tl.define_clonotypes(t_cells, receptor_arms="all", dual_ir="primary_only")

# visualizing the clonotype network
ir.tl.clonotype_network(t_cells, min_cells=2)
ir.pl.clonotype_network(t_cells, color="Sample", base_size=20, label_fontsize=9, panel_size=(7, 7))

# digging into clonotype expansion
ir.tl.clonal_expansion(t_cells)
t_cells.obs['clone_id'].replace(np.nan, 'NA', inplace=True)
sc.pl.umap(t_cells, color=["clone_id"])
ir.pl.clonal_expansion(t_cells, groupby="Sample", normalize=True, clip_at=5)

# viz UMAP by expansion (only color > 1)
exp_cmap = {'1': 'C7',
            '2': 'xkcd:cyan',
            '>= 3': 'xkcd:cyan',
            'nan': 'xkcd:light grey'}
sc.pl.umap(t_cells, color='clonal_expansion', palette=exp_cmap)

# clonotype abundance
ir.pl.group_abundance(t_cells, groupby="clone_id", target_col="Sample", max_cols=40)

# see % abundance for each clone
count_clone = np.zeros(len(t_cells.obs['clone_id'].unique()[1:]))
b = 0
for h in t_cells.obs['clone_id'].unique()[1:]:
    count_clone[b] = len(np.where(t_cells.obs['clone_id'] == h)[0])
    b = b + 1
