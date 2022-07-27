import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

# treatment palette
cmap = matplotlib.cm.get_cmap('viridis')
my_pal0 = {'YR': cmap(0.25),
           'YR_TTx': cmap(0.75)}

# fig params
sc.set_figure_params(figsize=(5, 5), fontsize=20)

# read in annotated file
new_bos = sc.read('/Users/katebridges/PycharmProjects/test/ttx_annotated20210211.h5ad')

# isolating myeloid cells only & rewriting PCs/UMAP
mdc_ind = np.concatenate((np.where(new_bos.obs['nn_80'] == 'Macrophage')[0],
                          np.where(new_bos.obs['nn_80'] == 'DC')[0], np.where(new_bos.obs['nn_80'] == 'Neutrophil')[0]))
mac_dc = new_bos[mdc_ind, :]

# cleaning up PCs for re-visualization


def remove_recompute(adata):
    del adata.obsm['X_umap'], adata.obsm['X_diffmap'], adata.obsm['X_pca'], adata.obsp, adata.uns
    sc.tl.pca(adata, svd_solver='auto')
    sc.pp.neighbors(adata)  # using with default parameters
    sc.tl.umap(adata)
    return adata


# VIZ of myeloid cells in separate UMAP space, by experimental condition
mac_dc = remove_recompute(mac_dc)
sc.pl.umap(mac_dc, legend_loc='lower right', color='Sample', s=40, palette=my_pal0)

# VIZ myeloid cells by cell type
my_pal1 = {'CD4+ T cell': cmap(0.01),
           'CD8+ T cell': cmap(0.5),
           'Treg': cmap(0.99)}
sc.pl.umap(mac_dc, legend_loc='lower right', color='nn_80', s=40, palette=my_pal1)
# using viridis instead of celltype_dict to maximally distinguish colors

# DIFF EXPR analysis
sample_celltype = []
for k in range(mac_dc.shape[0]):
    sample_celltype.append(mac_dc.obs['nn_80'][k] + ' ' + mac_dc.obs['Sample'][k])

mac_dc.obs['condID'] = pd.Categorical(sample_celltype)

# treated macs vs neuts vs dcs
sc.tl.rank_genes_groups(mac_dc, groupby='condID', group=['Macrophage YR_TTx', 'DC YR_TTx', 'Neutrophil YR_TTx'],
                        use_raw=True, key_added='Myeloid TTx')
mac_ttx_ = sc.get.rank_genes_groups_df(mac_dc, group='Macrophage YR_TTx', key='Myeloid TTx', pval_cutoff=0.01, log2fc_min=1.5)
dc_ttx = sc.get.rank_genes_groups_df(mac_dc, group='DC YR_TTx', key='Myeloid TTx', pval_cutoff=0.01, log2fc_min=1.5)
neut_ttx = sc.get.rank_genes_groups_df(mac_dc, group='Neutrophil YR_TTx', key='Myeloid TTx', pval_cutoff=0.01, log2fc_min=1.5)

# write DE results to xlsx
excelpath = '/Users/katebridges/PyCharmProjects/test/ttx_myeloid_20210820.xlsx'
writer = pd.ExcelWriter(excelpath, engine='xlsxwriter')

mac_ttx_.to_excel(writer, sheet_name='Mac TTx')
dc_ttx.to_excel(writer, sheet_name='DC TTx')
neut_ttx.to_excel(writer, sheet_name='Neutrophil TTx')

writer.save()  # for export to gProfiler for GSEA

# INDIV gene plots with bootstrapped error bars
matplotlib.rcParams.update({'font.size': 15})

bar_ind = ['DC Ctrl', 'DC TTx', 'Mac Ctrl', 'Mac TTx']
cc_geneset = ['Il12b', 'Il6', 'Tnf', 'Ccl5']  # genes of interest

for gene_name in cc_geneset:
    dat = np.array(mac_dc[:, gene_name].X.todense()).flatten()
    dat_stat = np.zeros((len(np.unique(mac_dc.obs['condID'])[:4]), 3))
    b = 0
    # excluding neutrophils for bootstrapping analyses
    for g in np.unique(mac_dc.obs['condID'])[:4]:
        i = np.where(mac_dc.obs['condID'] == g)[0]
        ci_info = bs.bootstrap(dat[i], stat_func=bs_stats.mean)
        dat_stat[b, 0] = ci_info.value
        dat_stat[b, 1] = dat_stat[b, 0] - ci_info.lower_bound
        dat_stat[b, 2] = ci_info.upper_bound - dat_stat[b, 0]
        b = b + 1

    # plotting results as bar graphs
    fig, ax = plt.subplots(figsize=(4, 5))
    barlist = ax.bar(bar_ind, dat_stat[:, 0], yerr=[dat_stat[:, 1], dat_stat[:, 2]], align='center', ecolor='black', capsize=10)
    barlist[0].set_color(sns.color_palette('tab20', 4)[3])
    barlist[1].set_color(sns.color_palette('tab20', 4)[2])
    barlist[2].set_color(sns.color_palette('tab20', 4)[1])
    barlist[3].set_color(sns.color_palette('tab20', 4)[0])
    plt.title(gene_name)
    ax.set_ylabel('ln[mRNA counts + 1]')
    ax.set_xticks(np.arange(len(bar_ind)))
    ax.set_xticklabels(bar_ind, rotation=75)
    plt.tight_layout()

# ENRICHMENT for DC subsets
dc_geneset = {'mregDC': ['Cd80', 'Cd40', 'Cd83', 'Cd86', 'Relb', 'Cd274', 'Pdcd1lg2', 'Cd200', 'Fas', 'Socs1',
                         'Socs2', 'Aldh1a2', 'Ccr7', 'Fscn1', 'Il4ra', 'Il4i1', 'Myo1g', 'Cxcl16', 'Adam8', 'Icam1',
                         'Marcks', 'Marcksl1'],
              'DC1': ['Xcr1', 'Clec9a', 'Cadm1', 'Naaa'],
              'DC2': ['Itgam', 'Cd209a', 'Sirpa', 'H2-DMb2']}

sc.tl.score_genes(mac_dc, dc_geneset['mregDC'], score_name='UCG_mregDC_score', use_raw=True)
sc.tl.score_genes(mac_dc, dc_geneset['DC1'], score_name='UCG_DC1_score', use_raw=True)
sc.tl.score_genes(mac_dc, dc_geneset['DC2'], score_name='UCG_DC2_score', use_raw=True)


def color_byscore(score, embed, embed_type, color_map, fig_size, title):
    fig, ax = plt.subplots(figsize=fig_size)
    scatter_x = embed[:, 0]
    scatter_y = embed[:, 1]
    fig1 = ax.scatter(scatter_x, scatter_y, c=score, s=4, cmap=color_map, vmin=-4, vmax=4)
    plt.colorbar(fig1)
    plt.title(title)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.xlabel(embed_type + '1')
    plt.ylabel(embed_type + '2')
    plt.show()


color_byscore(mac_dc.obs['UCG_mregDC_score'], mac_dc.obsm['X_umap'], 'UMAP', 'RdYlBu_r', (5, 4), 'UCG mregDC score')
color_byscore(mac_dc.obs['UCG_DC1_score'], mac_dc.obsm['X_umap'], 'UMAP', 'RdYlBu_r', (5, 4), 'UCG DC1 score')
color_byscore(mac_dc.obs['UCG_DC2_score'], mac_dc.obsm['X_umap'], 'UMAP', 'RdYlBu_r', (5, 4), 'UCG DC2 score')
