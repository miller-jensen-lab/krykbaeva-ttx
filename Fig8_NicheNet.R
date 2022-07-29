# example walkthrough for inferring ligands for DCs after TTx treatment
# loading necessary packages
library(dplyr)
library(Seurat) # v4 incoming?
library(ggplot2)
library(tidyr)
library(readr)
library(pheatmap)
library(tibble)
library(writexl)
library(anndata)
library(readxl)

# install.packages("devtools")
# devtools::install_github("saeyslab/nichenetr")

library(nichenetr)
library(tidyverse)

# reset working directory as applicable & read in scRNAseq (in RDS format)
ttx <- readRDS("/Users/katebridges/Downloads/opt3.rds")

# data has been preprocessed already/want to exclude imputed data for now
ttx@active.assay <- "RNA"

# need to add updated labels as metadata (from NN-based pipeline)
# read csv
nn80 <- read_excel('/Users/katebridges/Downloads/ttx_celltypelabels_20210322.xlsx')
ttx@meta.data$nn_80 <- nn80$nn_80

clust <- paste(ttx$nn_80, ttx@active.ident)

# reading in nichenet prior models
ligand_target_matrix = readRDS(url("https://zenodo.org/record/3260758/files/ligand_target_matrix.rds"))
lr_network = readRDS(url("https://zenodo.org/record/3260758/files/lr_network.rds"))
weighted_networks = readRDS(url("https://zenodo.org/record/3260758/files/weighted_networks.rds"))
weighted_networks_lr = weighted_networks$lr_sig %>% inner_join(lr_network %>% distinct(from,to), by = c("from","to"))

# converting nichenet prior models from human to mouse
lr_network = lr_network %>% mutate(from = convert_human_to_mouse_symbols(from), to = convert_human_to_mouse_symbols(to)) %>% drop_na()
colnames(ligand_target_matrix) = ligand_target_matrix %>% colnames() %>% convert_human_to_mouse_symbols()
rownames(ligand_target_matrix) = ligand_target_matrix %>% rownames() %>% convert_human_to_mouse_symbols()

ligand_target_matrix = ligand_target_matrix %>% .[!is.na(rownames(ligand_target_matrix)), !is.na(colnames(ligand_target_matrix))]

weighted_networks_lr = weighted_networks_lr %>% mutate(from = convert_human_to_mouse_symbols(from), to = convert_human_to_mouse_symbols(to)) %>% drop_na()

# perform the nichenet analysis - need to define sender & receiver populations
receiver = "DC"
Idents(object = ttx) <- 'nn_80'
expressed_genes_receiver = get_expressed_genes(receiver, ttx, pct = 0.10)

background_expressed_genes = expressed_genes_receiver %>% .[. %in% rownames(ligand_target_matrix)]

sender_celltypes = c("Macrophage","CD8+ T cell", "CD4+ T cell", "NK cell", "B cell", "Treg", 
                     "Fibroblast", "Tumor cell", 'Neutrophil')

list_expressed_genes_sender = sender_celltypes %>% unique() %>% lapply(get_expressed_genes, ttx, 0.10) # lapply to get the expressed genes of every sender cell type separately here

expressed_genes_sender = list_expressed_genes_sender %>% unlist() %>% unique()

# use DE to pull out genes of interest (sig upregulated after treatment)
seurat_obj_receiver= subset(ttx, idents = receiver)
seurat_obj_receiver = SetIdent(seurat_obj_receiver, value = seurat_obj_receiver[["Sample"]])

condition_oi = "YR_TTx"
condition_reference = "YR" 

DE_table_receiver = FindMarkers(object = seurat_obj_receiver, ident.1 = condition_oi, ident.2 = condition_reference, min.pct = 0.10) %>% rownames_to_column("gene")

geneset_oi = DE_table_receiver %>% filter(p_val_adj <= 0.1 & abs(avg_log2FC) >= 0.25) %>% pull(gene)
geneset_oi = geneset_oi %>% .[. %in% rownames(ligand_target_matrix)]

# defining potential ligands based on geneset of interest
ligands = lr_network %>% pull(from) %>% unique()
receptors = lr_network %>% pull(to) %>% unique()

expressed_ligands = intersect(ligands,expressed_genes_sender)
expressed_receptors = intersect(receptors,expressed_genes_receiver)

potential_ligands = lr_network %>% filter(from %in% expressed_ligands & to %in% expressed_receptors) %>% pull(from) %>% unique()

# perform nichenet analysis - ranking ligands by presence of target genes in geneset of interest
ligand_activities = predict_ligand_activities(geneset = geneset_oi, background_expressed_genes = background_expressed_genes, ligand_target_matrix = ligand_target_matrix, potential_ligands = potential_ligands)

ligand_activities = ligand_activities %>% arrange(-pearson) %>% mutate(rank = rank(desc(pearson)))

# visualizing best ligand results (keeping all)
best_upstream_ligands = ligand_activities %>% arrange(-pearson) %>% pull(test_ligand) %>% unique()

# inference & visualization of active target genes
active_ligand_target_links_df = best_upstream_ligands %>% lapply(get_weighted_ligand_target_links,geneset = geneset_oi, ligand_target_matrix = ligand_target_matrix, n = 200) %>% bind_rows() %>% drop_na()

active_ligand_target_links = prepare_ligand_target_visualization(ligand_target_df = active_ligand_target_links_df, ligand_target_matrix = ligand_target_matrix, cutoff = 0.33)

order_ligands = intersect(best_upstream_ligands, colnames(active_ligand_target_links)) %>% rev() %>% make.names()
order_targets = active_ligand_target_links_df$target %>% unique() %>% intersect(rownames(active_ligand_target_links)) %>% make.names()
rownames(active_ligand_target_links) = rownames(active_ligand_target_links) %>% make.names() # make.names() for heatmap visualization of genes like H2-T23
colnames(active_ligand_target_links) = colnames(active_ligand_target_links) %>% make.names() # make.names() for heatmap visualization of genes like H2-T23

vis_ligand_target = active_ligand_target_links[order_targets,order_ligands] %>% t()

p_ligand_target_network = vis_ligand_target %>% make_heatmap_ggplot("Prioritized ligands","Predicted target genes", color = "purple",legend_position = "top", x_axis_position = "top",legend_title = "Regulatory potential")  + theme(axis.text.x = element_text(face = "italic")) + scale_fill_gradient2(low = "whitesmoke",  high = "purple", breaks = c(0,0.0045,0.0090))
p_ligand_target_network

