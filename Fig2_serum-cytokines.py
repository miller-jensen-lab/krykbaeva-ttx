import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import rgb2hex
import seaborn as sns
from scipy.stats import ttest_ind

# reading in serum cytokine data (Eve tech)
filename = '/Users/katebridges/Downloads/Cytokines in sera in vivo - Bill Damsky_reformatted.xlsx'
data = pd.read_excel(filename)

# neglecting empty columns
data = data.iloc[:, :35]

# neglecting outlier values for now
cols = data.columns[1:]
mask = data[cols].applymap(lambda x: isinstance(x, (int, float)))
data[cols] = data[cols].where(mask)

# CALCULATING PCA with imputation for missing/OOR values
# need to address missing values - can drop or try imputation first
pc = PCA(data.iloc[:, 1:].T, missing='fill-em', ncomp=3)  # missing='fill-em', ncomp=2)

# VISUALIZATION of imputed data in clustered heatmap
sns.clustermap(pc.transformed_data, cmap='seismic', center=0, xticklabels=data['ProteinName'],
               yticklabels=data.columns[1:], linewidths=0.1, linecolor='black', rasterized=False)
plt.xticks(rotation=75)
plt.show()

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
plt.show()

# PLOTTING COEFFICIENTS ALONG PC (understanding drivers of variability in your dataset)
# horiz barplot for top 20 coeff along PC1
matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(7, 6.5))
plt.barh(np.arange(20), np.sort(-pc.coeff.iloc[0, :])[(45-20):], color='k')
plt.yticks(np.arange(20), data['ProteinName'][np.argsort(pc.coeff.iloc[0, :]).values[::-1]].values[(45-20):])
plt.xlabel('Coefficient along PC1')
plt.show()

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
                   split=True, linewidth=0.5, size=3, palette=sec_pal)
plt.xticks(ticks=np.arange(len(cc)), labels=cc)
plt.xticks(rotation=45)
plt.tight_layout()
# & remove legend
plt.legend([], [], frameon=False)

# associated statistical analysis - unpaired t test
mouse_stat = np.zeros((len(cc), 2))
b = 0
for r in cc:
    dat = dat_plot[r]
    mouse_stat[b, :] = ttest_ind(dat.values[:8][~np.isnan(dat.values[:8])],
                                 dat.values[8:][~np.isnan(dat.values[8:])])
    b = b + 1
