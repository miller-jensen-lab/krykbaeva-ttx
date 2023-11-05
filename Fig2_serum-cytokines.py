import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import rgb2hex
import seaborn as sns
from scipy.stats import ttest_ind
import click

@click.group()
def cli():
    pass

def read_in_data(filename: str):
    """
        The input file is going to looks something like the following:

        ```csv
        ProteinName   Healthy_1  Healthy_2  Healthy_3  Healthy_4  Healthy_5  Tumor d7_1  Tumor d7_2  Tumor d7_3  Tumor d7_4
        Eotaxin       1065.52    1700.03    1310.42    2002.69    1456.13    1356.29     1434.8      1498.08     1465.51
        G-CSF         6159.44    151.6      256.38     486.43     6563       456.15      687.56      468.59      375.33
        GM-CSF        20.34      29.98                33.93      42.93      27.1        16.13       16.13       16.13
        IFNg          5.45       4.8        1.6        30.67      2.88       11.19       6.73        4.8         6.73
        IL-1a         158.54     505.55     348.86     799.9      169.33     282.36      861.98      647.39      411.96
        IL-1b         16.3       94.22      59.31      42.03      48         50.9        45.04       53.76       16.3
        IL-2          2.53       9.6        4.99       94.39      6.9                   7.55        1.92        5.62
        ...
        ```
    """
    if filename.endswith('.csv'):
        data = pd.read_csv(filename)
    elif filename.endswith('.xlsx'):
        data = pd.read_excel(filename)
    else:
        raise ValueError('File must be either .csv or .xlsx')

    # neglecting empty columns
    data = data.iloc[:, :35]

    # neglecting outlier values for now
    cols = data.columns[1:]
    mask = data[cols].applymap(lambda x: isinstance(x, (int, float)))
    data[cols] = data[cols].where(mask)
    return data

@cli.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('--prefix', help='Prefix for output files', default='output')
@click.option('--show/--no-show', default=False)
def plot(filename: str, prefix: str, show: bool = False):
    """
        Plot heatmap and PCA for multiplexed serum cytokine data
    """
    data = read_in_data(filename)

    # CALCULATING PCA with imputation for missing/OOR values
    # need to address missing values - can drop or try imputation first
    pc = PCA(data.iloc[:, 1:].T, missing='fill-em', ncomp=3)  # missing='fill-em', ncomp=2)

    # VISUALIZATION of imputed data in clustered heatmap
    sns.clustermap(pc.transformed_data, cmap='seismic', center=0, xticklabels=data['ProteinName'],
                yticklabels=data.columns[1:], linewidths=0.1, linecolor='black', rasterized=False, z_score=1)
    plt.xticks(rotation=75)

    # Show and save
    if show:
        plt.show()
    plt.savefig(prefix + '_heatmap.png', dpi=300)

    # organizing indiv samples into groups by treatment
    cond_ind = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9])
    cond = np.array(['Healthy', 'Tumor d7', 'Tumor d9', 'PD-1', 'CD40', 'CSF1R', 'PD-1+CD40', 'PD-1+CSF1R', 'CD40+CSF1R', 'TTx'])

    # PLOTTING PCA with custom markers
    markers = ['o', 's', '^', 'v', 'D']
    cmap = matplotlib.cm.get_cmap('tab20b')
    cmap1 = matplotlib.cm.get_cmap('tab20c')

    my_pal = {0: cmap(0.875),
            1: cmap(0.925),
            2: cmap(0.975),
            3: cmap1(0.025),
            4: cmap(0.675),
            5: cmap1(0.075),
            6: cmap(0.725),
            7: cmap1(0.125),
            8: cmap1(0.225),
            9: cmap1(0.275)}

    # PLOTTING PCA WITH CUSTOM MARKERS
    fig, ax = plt.subplots()
    for g in np.unique(cond_ind):
        i = np.where(cond_ind == g)[0]
        if g < 5:
            ax.scatter(pc.factors['comp_0'][i], pc.factors['comp_2'][i], label=cond[g], marker=markers[0], s=24,
                    facecolors=my_pal[g], edgecolors=my_pal[g])
        else:
            ax.scatter(pc.factors['comp_0'][i], pc.factors['comp_2'][i], label=cond[g-5], marker=markers[0], s=24,
                    facecolors=my_pal[g], edgecolors=my_pal[g])
        # for h in i:
        #     ax.annotate(data.columns[h+1], (pc.factors['comp_0'][h], pc.factors['comp_2'][h]), c='black',
        #                 fontsize=8, ha='right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    plt.xlabel('PC1 ({}%)'.format(np.round((pc.eigenvals[0]/pc.eigenvals.sum())*100, 1)))
    plt.ylabel('PC3 ({}%)'.format(np.round((pc.eigenvals[2]/pc.eigenvals.sum())*100, 1)))

    if show:
        plt.show()
    plt.savefig(prefix + '_pca.png', dpi=300)

    # PLOTTING COEFFICIENTS ALONG PC (understanding drivers of variability in your dataset)
    # horiz barplot for top 20 coeff along PC1
    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(7, 6.5))
    plt.barh(np.arange(20), np.sort(-pc.coeff.iloc[0, :])[(45-20):], color='k')
    plt.yticks(np.arange(20), data['ProteinName'][np.argsort(pc.coeff.iloc[0, :]).values[::-1]].values[(45-20):])
    plt.xlabel('Coefficient along PC1')
    if show:
        plt.show()
    plt.savefig(prefix + '_pc1.png', dpi=300)

    # for PC3
    matplotlib.rcParams.update({'font.size': 18})
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    plt.barh(np.arange(20), np.sort(-pc.coeff.iloc[2, :])[(45-20):], color='k')
    plt.yticks(np.arange(20), data['ProteinName'][np.argsort(pc.coeff.iloc[2, :]).values[::-1]].values[(45-20):])
    plt.xlabel('Coefficient along PC3')
    if show:
        plt.show()
    plt.savefig(prefix + '_pc3.png', dpi=300)

    # FOR CD40 INCLUSIVE SAMPLES ONLY (Panel E)
    new_df = data.iloc[:, 1:].T
    cd40_samples = new_df[new_df.index.str.contains('CD40') | new_df.index.str.contains('TTx')]
    cd40_samples.columns = data['ProteinName']

    ctrl_means = new_df[new_df.index.str.contains('Tumor')].mean(axis=0)
    csf1r_means = new_df[new_df.index.str.contains('CSF1R')].iloc[:6, :].mean(axis=0)
    cd40_means = new_df[new_df.index.str.contains('CD40')].iloc[:6, :].mean(axis=0)
    ttx_means = new_df.iloc[-6:, :].mean(axis=0)

    logfc = pd.DataFrame([])
    logfc['CSF1R'] = np.log2(csf1r_means/ctrl_means)
    logfc['CD40'] = np.log2(cd40_means/ctrl_means)
    logfc['CD40+CSF1R'] = np.log2(ttx_means/ctrl_means)
    logfc.index = data['ProteinName']

    # drop rows with nan
    logfc = logfc.dropna(axis=0)

    sns_plot = sns.clustermap(logfc, cmap='seismic', center=0, linewidths=0.1, linecolor='black', rasterized=False,
                            yticklabels=logfc.index, figsize=(6, 14))
    sns_plot.savefig(prefix + '_log2fc_cd40_csf1r.png', dpi=300)

    # cytokines/chemokines only
    main = ['TNFa', 'IFNg', 'CXCL9', 'IL-12p40', 'IL-6', 'CCL5', 'CXCL10']
    supp_gf = ['MCP-1', 'CCL3', 'CCL4', 'CXCL1', 'CCL17', 'CCL22', 'CXCL2', 'CX3CL1', 'G-CSF', 'LIF', 'M-CSF', 'VEGF']

    sns_plot = sns.clustermap(logfc.loc[supp_gf], cmap='seismic', center=0, linewidths=0.1, linecolor='black', rasterized=False,
                            yticklabels=supp_gf, figsize=(8, 5), row_cluster=False, col_cluster=False)

    # # remove CCL21
    # cd40_samples = cd40_samples.drop(columns='CCL21')
    #
    # pc_cd40 = PCA(cd40_samples, missing='fill-em', ncomp=3)
    # cd40_cond = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    #
    # fig, ax = plt.subplots()
    # for n in np.unique(cd40_cond):
    #     i = np.where(cd40_cond == n)[0]
    #     ax.scatter(pc_cd40.factors['comp_0'][i], pc_cd40.factors['comp_1'][i], label=n)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    # plt.xlabel('PC1 ({}%)'.format(np.round((pc_cd40.eigenvals[0]/pc.eigenvals.sum())*100, 1)))
    # plt.ylabel('PC2 ({}%)'.format(np.round((pc_cd40.eigenvals[1]/pc.eigenvals.sum())*100, 1)))
    #
    # matplotlib.rcParams.update({'font.size': 18})
    # fig, ax = plt.subplots(figsize=(5, 8))
    # plt.bar(np.arange(44), np.sort(pc_cd40.coeff.iloc[0, :]), color='k')
    # plt.xticks(np.arange(44), cd40_samples.columns[np.argsort(pc_cd40.coeff.iloc[0, :])], rotation=90)
    # plt.ylabel('Coefficient along PC1')
    # plt.show()

    # FOR FIG 3:  creating similar comparison to human data (untreated tumors vs. either CD40+CSF1R or TTx)
    human_cond = np.array([])
    for j in ['Tumor d7', 'Tumor d9', 'CD40\+CSF1R', 'TTx']:
        human_cond = np.concatenate((human_cond, np.where(data.columns.str.contains(j))[0]))

    data_plot = data.iloc[:, 1:].T
    data_plot.columns = data['ProteinName']

    cc = ['M-CSF', 'CXCL10', 'CCL22', 'MCP-1', 'TNFa', 'IFNg', 'CCL4', 'Eotaxin', 'IL-12p40', 'CCL17', 'IL-10',
        'G-CSF', 'CCL3', 'IL-6', 'GM-CSF', 'IL-15', 'IL-7']

    dat_plot = data_plot[cc].iloc[human_cond-1, :]

    x_ind = np.array([])
    for k in range(dat_plot.shape[1]):
        x_ind = np.concatenate((x_ind, np.tile(k, dat_plot.shape[0])))

    time_lab = np.concatenate((np.tile(0, 8), np.tile(1, 6)))

    # VISUALIZING CERTAIN CYTOKINES, ETC WITH BOXPLOTS (unpaired T test for statistical analysis follows)
    human_plot_df = pd.DataFrame({'Panel': x_ind,
                                'Concentration [ln(pg/mL)]': np.log(dat_plot.values.flatten('F')+1),
                                'Treatment': np.tile(time_lab, dat_plot.shape[1])})

    sec_pal = {0: 'xkcd:cerulean',
            1: 'xkcd:light red'}

    fig, ax = plt.subplots(figsize=(4, 18))
    sns.boxplot(x="Panel", y="Concentration [ln(pg/mL)]", hue="Treatment",
                data=human_plot_df, fliersize=0, width=0.8, palette=sec_pal)
    sp = sns.stripplot(x="Panel", y="Concentration [ln(pg/mL)]", hue="Treatment",
                    data=human_plot_df, jitter=True,
                    # This option appears to have been removed
                    # since Kate wrote this code.
                    # split=True,
                    linewidth=0.5, size=3, palette=sec_pal)
    plt.xticks(ticks=np.arange(len(cc)), labels=cc)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # & remove legend
    plt.legend([], [], frameon=False)

    # KLJ: Why is this not saved?

    # associated statistical analysis - unpaired t test
    mouse_stat = np.zeros((len(cc), 2))
    b = 0
    for r in cc:
        dat = dat_plot[r]
        values_left = dat.values[:8][~np.isnan(dat.values[:8])]
        values_right = dat.values[8:][~np.isnan(dat.values[8:])]
        mouse_stat[b, :] = ttest_ind(values_left, values_right)
        b = b + 1

    # KLJ: Doesn't look like this is used anywhere


if __name__ == '__main__':
    """
        filename = '/Users/katebridges/Downloads/Cytokines in sera in vivo - Bill Damsky_reformatted.xlsx'
    """
    cli()