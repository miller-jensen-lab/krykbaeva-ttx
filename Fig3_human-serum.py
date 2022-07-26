import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import rgb2hex
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_rel

# need to read in data & relevant attributes
filename = '/Users/katebridges/Downloads/ObsConc_68analytes_OOR.xlsx'
fileattr = '/Users/katebridges/Downloads/Samples.xlsx'

data = pd.read_excel(filename)
attr = pd.read_excel(fileattr)
attr.replace(np.nan, 'NA', inplace=True)

patients = []
for q in attr['patient and sample']:
    patients.append(q.split(' ', 1)[0])

attr['patient'] = patients

# initial exclusion of cohort 4 patient #4 (elevated C/C across the board)
data = data.iloc[np.where(attr['patient and sample'].str.contains('0404') == False)[0], :]
attr = attr.iloc[np.where(attr['patient and sample'].str.contains('0404') == False)[0], :]

# inspecting melanoma patients only - sans 404
mel_ind = np.where(attr['Cancer type'] == 'Melanoma')[0]
data_mel = data.iloc[mel_ind, :]
attr_mel = attr.iloc[mel_ind, :]

# only considering cycle 1 data for now
mel_c1 = data_mel.iloc[np.where(attr_mel['Cycle/Day'].str.contains('C1'))[0], :]
attrmel_c1 = attr_mel.iloc[np.where(attr_mel['Cycle/Day'].str.contains('C1'))[0], :]

# bar plots & statistical analysis to figure out which C/C are signfiicantly upregulated @ C1D2 compared to baseline
human_stat = np.zeros((len(mel_c1.columns[1:]), 2))
b = 0
baseline = np.where(attrmel_c1['patient and sample'].str.contains('baseline'))[0]
post_c1 = np.where(attrmel_c1['patient and sample'].str.contains('24'))[0]
for r in mel_c1.columns[1:]:
    dat = mel_c1[r]
    human_stat[b, :] = ttest_rel(dat.values[baseline], dat.values[post_c1])
    b = b + 1

# identifying C/C significantly different 24 hr after therapy
sigcc_pd = pd.DataFrame(data=human_stat, index=mel_c1.columns[1:], columns=['statistic', 'pval'])
sig_cc = sigcc_pd.iloc[np.where(human_stat[:, 1] < 0.05)[0], :]

# PLOTTING C/C to match with mouse data
cc = ['M-CSF', 'CXCL10', 'CCL22', 'CCL2', 'TNFa', 'IFNg', 'CCL4', 'CCL11', 'IL-12P40', 'CCL17', 'IL-10',
      'G-CSF', 'CCL3', 'IL-6', 'GM-CSF', 'IL-15', 'IL-7']

c1_rearr = mel_c1[cc].iloc[np.concatenate((baseline, post_c1)), :]
attrc1_rearr = attrmel_c1.iloc[np.concatenate((baseline, post_c1)), :]

x_ind = np.array([])
for k in range(c1_rearr.shape[1]):
    x_ind = np.concatenate((x_ind, np.tile(k, c1_rearr.shape[0])))

time_lab = np.concatenate((np.tile(0, baseline.shape[0]), np.tile(1, post_c1.shape[0])))

human_plot_df = pd.DataFrame({'Panel': x_ind,
                              'Concentration [ln(pg/mL)]': np.log(c1_rearr.values.flatten('F')+1),
                              'C1 Timepoint': np.tile(time_lab, c1_rearr.shape[1])})

sec_pal = {0: 'xkcd:cerulean',
           1: 'xkcd:light red'}

# fig, ax = plt.subplots(figsize=(4, 4.5))
sns.boxplot(x="Panel", y="Concentration [ln(pg/mL)]", hue="C1 Timepoint",
            data=human_plot_df, fliersize=0, width=0.8, palette=sec_pal)
sp = sns.stripplot(x="Panel", y="Concentration [ln(pg/mL)]", hue="C1 Timepoint",
                   data=human_plot_df, jitter=True,
                   split=True, linewidth=0.5, size=3, palette=sec_pal)
plt.xticks(ticks=np.arange(len(cc)), labels=cc)
plt.xticks(rotation=45)
plt.tight_layout()
# & remove legend
plt.legend([], [], frameon=False)

# PCA ANALYSIS & PLOTTING
pc = PCA(mel_c1.iloc[:, 1:])

# let's start by looking at all combinations of first 5 PCs - variance along PC1 comes from primary outlier (#0404)
fig_dir = '/Users/katebridges/PycharmProjects/test/PCA_melanoma/'

fig, ax = plt.subplots(figsize=(4.25, 4))
plt.subplots_adjust(left=0.2)
b = 0
for g in np.unique(attrmel_c1['CD40 Dose']):
    j = np.where(attrmel_c1['CD40 Dose'] == g)[0]
    for d in j:
        ax.scatter(pc.factors['comp_00'].values[d], -pc.factors['comp_01'].values[d], s=24, label=g,
                   edgecolors=rgb2hex(sns.husl_palette(3)[b]), facecolors='none')
    b = b + 1
ax.legend(loc='upper left', prop={'size': 8})
plt.xlabel('PC1 ({}%)'.format(np.round((pc.eigenvals[0]/pc.eigenvals.sum())*100, 1)))
plt.ylabel('PC2 ({}%)'.format(np.round((pc.eigenvals[1]/pc.eigenvals.sum())*100, 1)))
plt.title('CD40 Dose')
plt.savefig(fig_dir + 'CD40.png', dpi=150), plt.close(fig)

# COEFF for each C/C on panel along PC2
matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(7, 6.25))
plt.subplots_adjust(bottom=0.25)
plt.barh(np.arange(mel_c1.iloc[:, 1:21].shape[1]), np.sort(-pc.coeff.iloc[1, :])[(68-20):], color='k')
plt.yticks(np.arange(mel_c1.iloc[:, 1:21].shape[1]), data.columns[1:][np.argsort(pc.coeff.iloc[1, :]).values[::-1]][(68-20):])
plt.xlabel('Coeff along PC2')
plt.ylim([-1, 20])
plt.tight_layout()
plt.savefig('PC2_flip.png', dpi=150), plt.close(fig)


# INSPECTING CORRELATIONS among C/C panel - seeing which cytokines are "moving together" in response to therapy
corr_data = np.zeros((data.shape[1] - 1, data.shape[1] - 1))
for k in np.arange(1, data.shape[1]):
    for j in np.arange(1, data.shape[1]):
        # if k >= j:
        corr_data[k-1, j-1] = pearsonr(mel_c1.iloc[:, k], mel_c1.iloc[:, j])[0]

sns.set(font_scale=0.5)
sg = sns.clustermap(corr_data, yticklabels=data.columns[1:], xticklabels=data.columns[1:], cmap='viridis',
                    linewidths=0.1, linecolor='black', rasterized=False)
plt.xticks(rotation=75)
plt.show()

# limiting correlation heatmap to C/C with changes in response to therapy
clustermap_order = []
for r in np.arange(len(sg.ax_heatmap.yaxis.get_majorticklabels())):
    clustermap_order.append(sg.ax_heatmap.yaxis.get_majorticklabels()[r].get_text())

corr_data = np.zeros((len(clustermap_order[:32]), len(clustermap_order[:32])))
for k in np.arange(len(clustermap_order[:32])):
    for j in np.arange(len(clustermap_order[:32])):
        # if k >= j:
        corr_data[k, j] = pearsonr(mel_c1[clustermap_order[k]], mel_c1[clustermap_order[j]])[0]

sns.set(font_scale=1.25)
sg = sns.clustermap(corr_data, yticklabels=clustermap_order, xticklabels=clustermap_order, cmap='viridis',
                    linewidths=0.1, linecolor='black', rasterized=False)
plt.xticks(rotation=75)
plt.show()
sg.savefig('pairedcorr_limited.png')
