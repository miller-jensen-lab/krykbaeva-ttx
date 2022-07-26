import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import phenograph
import umap
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import multiprocessing as mp
mp.set_start_method("fork")

# read in the data
tam = pd.read_csv('/Users/katebridges/Downloads/Chip 1 Untreated TAM OnlyON.csv')
tadc = pd.read_csv('/Users/katebridges/Downloads/Chip 2 Untreated TADC OnlyON.csv')
tam_t = pd.read_csv('/Users/katebridges/Downloads/Chip 3 Treated TAM OnlyON.csv')
tadc_t = pd.read_csv('/Users/katebridges/Downloads/Chip 4 Treated TADC OnlyON.csv')

thresh = np.arcsinh(1/0.8)

# exclude CXCL9/10 (poor antibodies)
corr_ind = np.array([15, 16, 4, 13, 14, 10, 12, 17, 7, 11, 3, 5, 6])

tam_df = pd.concat([tadc.iloc[:, corr_ind], tadc_t.iloc[:, corr_ind], tam.iloc[:, corr_ind], tam_t.iloc[:, corr_ind]])  # tam.iloc[:, 3:], tam_t.iloc[:, 3:]]


x_ind = np.array([])
for k in range(tam_df.shape[1]):
    x_ind = np.concatenate((x_ind, np.tile(k, tam_df.shape[0])))
tam_cat_lab = np.concatenate((np.tile(0, tadc.shape[0]), np.tile(1, tadc_t.shape[0]), np.tile(2, tam.shape[0]), np.tile(3, tam_t.shape[0])))

# create dataframe for plotting
tam_plot_df = pd.DataFrame({
    'Panel': x_ind,
    'Signal intensity (a.u.)': tam_df.values.flatten('F'),
    'Treatment condition': np.tile(tam_cat_lab, tam_df.shape[1])
})

# using bootstrapping to construct mean +/- errors bars for each sc-sec measurement
dat_stat = np.zeros((tam_df.shape[1], 3, len(np.unique(tam_cat_lab))))
for h in np.unique(tam_cat_lab):
    i = np.where(tam_cat_lab == h)[0]
    for g in np.arange(tam_df.shape[1]):
        ci_info = bs.bootstrap(tam_df.iloc[i, g].to_numpy(), stat_func=bs_stats.mean)
        dat_stat[g, 0, h] = ci_info.value
        dat_stat[g, 1, h] = ci_info.value - ci_info.lower_bound
        dat_stat[g, 2, h] = ci_info.upper_bound - ci_info.value

# PLOTTING sc-sec data as boxplots for TADCs alone
sec_pal = {0: sns.color_palette('tab20', 4)[3],
           1: sns.color_palette('tab20', 4)[2],
           2: sns.color_palette('tab20', 4)[1],
           3: sns.color_palette('tab20', 4)[0]}

fig, ax = plt.subplots(figsize=(16, 3.5))
sns.boxplot(x="Panel", y="Signal intensity (a.u.)", hue="Treatment condition",
            data=tam_plot_df.iloc[np.where(tam_plot_df['Treatment condition'] < 2)[0], :], fliersize=0, width=0.8,
            palette=sec_pal)
sp = sns.stripplot(x="Panel", y="Signal intensity (a.u.)", hue="Treatment condition",
                   data=tam_plot_df.iloc[np.where(tam_plot_df['Treatment condition'] < 2)[0], :], jitter=True,
                   split=True, linewidth=0.5, size=3, palette=sec_pal)

plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], labels=tam_df.columns)

handles, labels = ax.get_legend_handles_labels()
L = ax.legend(handles[0:4], labels[0:4], handletextpad=0.5, bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)
L.get_texts()[0].set_text('TADC')
L.get_texts()[1].set_text('TADC_T')
# L.get_texts()[2].set_text('TAM')
# L.get_texts()[3].set_text('TAM_T')

# plotting percentage above threshold
x1 = -0.5
x2 = 12.5
y1 = thresh
y2 = y1
plt.plot([x1, x1, x2, x2], [y1, y2, y2, y1], linewidth=1, color='k')
g = 0
for b in tam_df.columns:
    dat0 = tadc[b]
    dat1 = tadc_t[b]
    plt.text(g-0.35, -1.75, '{}%'.format(np.round(100 * len(np.where(dat0 > thresh)[0]) / len(dat0), 1)), fontsize=8, color=sns.color_palette('tab20', 4)[3])
    plt.text(g+0.1, -1.85, '{}%'.format(np.round(100 * len(np.where(dat1 > thresh)[0]) / len(dat1), 1)), fontsize=8, color=sns.color_palette('tab20', 4)[2])
    g = g + 1
plt.show()

# CLUSTERING & 2D EMBEDDING
# PhenoGraph clustering
communities, graph, Q = phenograph.cluster(tam_df.values, k=150, n_jobs=1)

# visualizing cluster definitions by secretion in a heatmap
scores = np.zeros((len(np.unique(communities)), tam_df.shape[1]))
for i in np.unique(communities):
    ind = np.where(communities == i)[0]
    group = tam_df.iloc[ind, :]
    for j in range(tam_df.shape[1]):
        scores[i, j] = 100*len(np.where(group.iloc[:, j] > np.arcsinh(1/0.8))[0])/len(group.iloc[:, j])

fig, (cax, ax) = plt.subplots(nrows=2, figsize=(7,4.025),  gridspec_kw={"height_ratios":[0.025, 1]})
hm = sns.heatmap(scores, ax=ax, cmap='viridis', xticklabels=tam_df.axes[1],
                 yticklabels=['{} ({}%)'.format(p, np.around(100*len(np.where(communities == p)[0])/tam_df.shape[0], decimals=2)) for p in np.unique(communities)],
                 cbar=False)
plt.xticks(rotation=75)
plt.yticks(rotation=0)
plt.subplots_adjust(bottom=0.2, left=0.2)
fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
plt.show()


def plot_phenograph(embed, clustering, embed_type, fig_size, title, cmap):
    fig, ax = plt.subplots(figsize=fig_size)
    scatter_x = embed[:, 0]
    scatter_y = embed[:, 1]
    for g in np.unique(clustering):
        i = np.where(clustering == g)
        ax.scatter(scatter_x[i], scatter_y[i], label=g, s=8, c=cmap[g])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.xlabel(embed_type + '1')
    plt.ylabel(embed_type + '2')
    plt.title(title)
    plt.subplots_adjust(right=0.8)
    # plt.xlim([4, 14])
    # plt.ylim([3, 12])
    plt.show()


# PLOTTING cluster labels over 2D UMAP embedding
color_map = sns.color_palette('tab20', 4)

cmap_dict = {0: color_map[3],
             1: color_map[2],
             2: color_map[1],
             3: color_map[0],
             }
embedding = umap.UMAP().fit_transform(tam_df)
plot_phenograph(embedding, communities, 'UMAP', (4.5, 4), '', sns.color_palette('husl', 4))
