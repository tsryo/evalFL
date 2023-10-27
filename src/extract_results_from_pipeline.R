# Author: T.Y.
# Date created: 19.06.2023

# Script used to gather and analyze saved models and predictions from model train/testing in 'pipeline_train_test_save.R'


##### library imports #####
# set current dir to one-level above current script dir
setwd(sprintf("%s/..",dirname(rstudioapi::getSourceEditorContext()$path)))
# assumes commons repo is cloned locally in same directory as current repo https://github.com/tsryo/commons
source("../commons/src/orchestrator.R")
source("src/fl_orchestrator.R")
library(ggplot2)
library(gridExtra)
library(grid)
library(dplyr)
library(pROC)
library(dplyr)
library(igraph)


cat("\n\n********************************************************************\n\n")
cat("\t\t\t Loaded libraries \n")
cat("\n\n********************************************************************\n\n")


##### config [modify as per usage] #####
N_FOLDS_CV = 10

RUN_INIT_DFS = F # should results dataframes be (re)initialized? (if running for first time with new results, yes)
RUN_INIT_per_record_preds = F # only considered when RUN_INIT_POOLED_RES == T
RUN_INIT_POOLED_RES = F # only considered when RUN_INIT_POOLED_RES == T
RUN_INIT_NRI_MATRIX = F # only considered when RUN_INIT_POOLED_RES == T

AUC_TEST_SIG = F # run test for AUC-ROC significant differences between (paired) model predictions?
RUN_REMA_MEANS = F # run random-effects meta-analysis (with center as random effect) pooling for performance metrics?
SAVE_XLSX_per_center_metrics = F # as is written on the tin
GEN_PLOTS = F # generate calibration-graphs and AUC-ROC curve graphs?
GEN_VARS_SELECTED = F # dataframe in long format with all variables selected per model setup
SAVE_NRI_MATRIX = F   # containing net reclassificaiton improvement results
RUN_DOT_PLOT_MODEL_PREDS = F # for inspecting model predictions from two models against each-other


RFILES = list(
  "LCOA"  = list(
    "global" = "../data/out/lr-lasso-global--15.RData",
    "local" = -1, # no LCOA for local model strategy
    "fedavg_no_recal" =   "../data/out/fed-lr-lasso-gradient-update--37.RData",
    "fedavg_glob_recal" =   "../data/out/fed-lr-lasso-gradient-update--39.RData",
    "ensemble_no_recal" = "../data/out/fed-lr-lasso-ensemble--19.RData" ,
    "ensemble_glob_recal"  = "../data/out/fed-lr-lasso-ensemble--21.RData"
  ),
  "CV" = list(
    "global" = "../data/out/lr-lasso-global--14.RData",
    "local" = "../data/out/lr-lasso-locals--12.RData",
    "fedavg_no_recal" = "../data/out/fed-lr-lasso-gradient-update--36.RData",
    "fedavg_glob_recal" =   "../data/out/fed-lr-lasso-gradient-update--38.RData",
    "ensemble_no_recal" = "../data/out/fed-lr-lasso-ensemble--16.RData" ,
    "ensemble_glob_recal"  = "../data/out/fed-lr-lasso-ensemble--20.RData" # 19.07.2023
  )
)

SENS_ANALYSIS_FN_PREFIX = '' # add prefix to results files saved (for easier distinction between main and sensitivity analysis results files)
# low-volume centers excluded
# SENS_ANALYSIS_FN_PREFIX = 'excl_c_low_volume'
RFILES_sens_analysis1 = list(
  "LCOA"  = list(
    "global" = "../data/out/lr-lasso-global--18.RData",
    "local" = -1,
    "fedavg_no_recal" =   "../data/out/fed-lr-lasso-gradient-update--43.RData",
    "fedavg_glob_recal" = "../data/out/fed-lr-lasso-gradient-update--29.RData",
    "ensemble_no_recal" = "../data/out/fed-lr-lasso-ensemble--25.RData",
    "ensemble_glob_recal"  = "../data/out/fed-lr-lasso-ensemble--24.RData"
  ),
  "CV" = list(
    "global" = "../data/out/lr-lasso-global--16.RData",
    "local" = "../data/out/lr-lasso-locals--14.RData",
    "fedavg_no_recal" = "../data/out/fed-lr-lasso-gradient-update--42.RData" ,
    "fedavg_glob_recal" = "../data/out/fed-lr-lasso-gradient-update--28.RData",
    "ensemble_no_recal" = "../data/out/fed-lr-lasso-ensemble--23.RData",
    "ensemble_glob_recal"  = "../data/out/fed-lr-lasso-ensemble--22.RData"
  )
)
RFILES_sens_analysis1$readme = 'low-volume centers excluded'
# RFILES = RFILES_sens_analysis1 # example override of input model files to use for sens analysis results generation

# Preserve anonymity of center codes given
inv_hidden_center_mapping = list(
  "A" = "1" ,
  "B" = "2" ,
  "C" = "3" ,
  "D" = "4" ,
  "E" = "5" ,
  "F" = "6" ,
  "G" = "7" ,
  "H" = "8" ,
  "I" = "9" ,
  "J" = "10" ,
  "K" = "11" ,
  "L" = "12" ,
  "M" = "13" ,
  "N" = "14" ,
  "O" = "15" ,
  "P" = "16",
  "0ALL" = "ALL")

hidden_center_mapping = list("1" = "A",
                             "2" = "B",
                             "3" = "C",
                             "4" = "D",
                             "5" = "E",
                             "6" = "F",
                             "7" = "G",
                             "8" = "H",
                             "9" = "I",
                             "10" = "J",
                             "11" = "K",
                             "12" = "L",
                             "13" = "M",
                             "14" = "N",
                             "15" = "O",
                             "16" = "P",
                             "ALL" = "0ALL")






###### functions #####

#' Generate ordering of NRI results using a directed (acyclical, when results are consistent) graph
nri_dag_ordering <- function(nri_matrix, ignore_non_sig = F, alt_formula_nri = T, verbose = F) {
  nri_matrix_means = nri_matrix[-grep("_.b",rownames(nri_matrix)),] # remove upper/lower bound values
  # compute statements from NRI values, like model1 > model2 (sig/non-sig)
  statements = list()
  for(r_idx in 1:(nrow(nri_matrix_means)-1)) {
    for(c_idx in (r_idx+1):ncol(nri_matrix_means)) {
      c_m1 = rownames(nri_matrix_means)[r_idx]
      c_m2 = cns(nri_matrix_means)[c_idx]
      c_nri = nri_matrix_means[r_idx, c_idx]
      c_nri_lb = nri_matrix[sprintf("%s_lb", c_m1), c_m2]
      c_nri_ub = nri_matrix[sprintf("%s_ub", c_m1), c_m2]
      middle_nri_value = if(alt_formula_nri) 0 else 0.5
      is_better = c_nri > middle_nri_value
      is_sig = (is_better && c_nri_lb > middle_nri_value) || (!is_better && c_nri_ub < middle_nri_value)
      statements[[len(statements)+1]] = if(is_better) c(c_m1, c_m2) else c(c_m2, c_m1)
      if(!is_sig)
        statements[[len(statements)]] = c(statements[[len(statements)]], "non-sig")
      if(!is_sig && ignore_non_sig)
        statements[[len(statements)]] = NULL
      if(is_sig && verbose) {
        print(sprintf("%s %s %s%s", c_m1, if(is_better) ">" else "<", c_m2, if(is_sig) "*" else ""))
      }
    }
  }
  return(generate_possible_orderings(statements, cns(nri_matrix_means)))
}

#' Function to generate all possible orderings (used for NRI matrixes)
#' # # Example usage:
# variables <- c("A", "B", "C", "D")
#
# statements <- list(
#   c("A", "C"),
#   c("C", "B"),
#   c("D", "A")
# )
#
# possible_orderings <- generate_possible_orderings(statements, variables)
# print(possible_orderings)
generate_possible_orderings <- function(statements, variables) {
  # Create an empty directed graph
  graph <- graph.empty(directed = TRUE)

  # Add nodes (variables) to the graph
  graph <- add.vertices(graph, name = variables, nv = len(variables))

  # Add directed edges based on the statements
  for (stmt in statements) {
    graph <- add_edges(graph, c(which(variables ==stmt[1]), which(variables ==stmt[2])) , color = if(len(stmt) > 2) "red" else "gray")
  }
  plot(graph)

  # Check for cycles in the graph
  if (is_dag(graph)) {
    # Perform topological sort to get a possible ordering
    orderings <- topo_sort(graph)
    mtext(paste0(names(orderings), collapse = ' '), side = 3)
    return(list(orderings))
  } else {
    # If the graph contains cycles, return an empty list
    return(list())
  }
}

#' c_fn - saved model file from running train/test pipeline.
classify_results_file <- function(c_fn, verbose = T) {
  c_res = load_with_assign(c_fn)

  is_on_full_dataset = nrow(c_res$preds_df) == 33322 #  [16661 * 2] including final model
  deviation_from_full_dataset = 33322 - nrow(c_res$preds_df)
  f_table = table(c_res$preds_df$fold)
  is_LCOA = sd(f_table[1:(len(f_table)-1)]) > 100
  is_CV = !is_LCOA

  missing_folds = c()
  missing_centers = c()
  if(!is_on_full_dataset){
    if(is_CV)
      missing_folds = setdiff(1:11, as.numeric(names(table(c_res$preds_df$fold))))
    else
      missing_centers = setdiff(1:17, as.numeric(names(table(c_res$preds_df$fold))))
  }

  is_fedavg = "model-update-transmission" %in% class(c_res$fitted_models[[1]])
  all_agsls_same = NA
  all_lrs_same = NA
  all_lrs = c()
  all_agsls = c()

  if(is_fedavg){
    all_lrs = umap( 1:len(c_res[["fitted_models"]]), function(x) { c_res[["fitted_models"]][[x]][["training_params"]][["learning_rate"]]} )
    all_lrs_same = len(uniq(all_lrs)) == 1
    all_agsls = umap( 1:len(c_res[["fitted_models"]]), function(x) { c_res[["fitted_models"]][[x]][["training_params"]][["agrreement_strength_lasso"]]} )
    all_agsls_same = len(uniq(all_agsls)) == 1
  }
  c_int_meta_res = get_mean_metric_from_result_obj(c_res, metric_nm = 'calibration-intercept')
  na_cints = which(is.na(c_int_meta_res$individual_means))

  auc_meta_res = get_mean_metric_from_result_obj(c_res, metric_nm = 'AUC-ROC')
  na_aucs = which(is.na(auc_meta_res$individual_means))

  res = list( filename = c_fn,
              val_strat = if(is_CV) "CV" else "LCOA",
              is_on_full_dataset = if(is_on_full_dataset) "YES" else "NO",
              cal_intercepts = try_round(sort(c_int_meta_res$individual_means, na.last = T), 2),
              aucs = try_round(sort(auc_meta_res$individual_means, na.last = T), 2))

  if(len(na_aucs) != 0)
    res$na_aucs = paste0(na_aucs, collapse = ', ')
  if(len(na_cints) != 0)
    res$na_cints = paste0(na_cints, collapse = ', ')

  if(!is_on_full_dataset) {
    res$deviation_from_full_dataset = deviation_from_full_dataset
    if(is_CV)
      res$missing_folds = paste0(missing_folds, collapse = ', ')
    else
      res$missing_centers = paste0(missing_centers, collapse = ', ')
  }

  if(is_fedavg) {
    res$all_lrs_same = if(all_lrs_same) sprintf("YES [%s]", all_lrs[1]) else "NO"
    res$all_agsls_same = if(all_agsls_same) sprintf("YES [%s]", all_agsls[1]) else "NO"
    res$lrs = all_lrs
    res$agsls = all_agsls
  }

  if(verbose)
    for(c_k in names(res))
      cat(sprintf("%s = %s\n", c_k, if(len(res[[c_k]]) > 1) paste0(res[[c_k]], collapse = ', ') else res[[c_k]]))

  return(res)

}

add_to_all_preds_dfs_cur_res <- function(c_fn, val_strat, model_tp, all_preds_df, all_final_preds_df) {
  res_df = load_with_assign(c_fn)
  preds_df = res_df$preds_df
  final_preds_df = preds_df[preds_df$fold == max(preds_df$fold), ] # final model preds only
  final_preds_df = final_preds_df[,c("prediction", "outcome", "center", "model_name", "fold", 'row_id')]
  final_preds_df$val_strat = val_strat
  final_preds_df$MDS = model_tp
  all_final_preds_df = rbind(all_final_preds_df, final_preds_df)

  preds_df = preds_df[preds_df$fold != max(preds_df$fold), ] # remove final model results
  preds_df = preds_df[,c("prediction", "outcome", "center", "model_name", "fold", 'row_id')]
  preds_df$val_strat = val_strat
  preds_df$MDS = model_tp
  all_preds_df = rbind(all_preds_df, preds_df)
  return(list(all_preds_df = all_preds_df, all_final_preds_df = all_final_preds_df))
}

add_all_preds_from_RFILES_list <- function(RFILES) {
  all_preds_df = NULL
  all_final_preds_df = NULL
  for(val_strat in names(RFILES)) {
    for(model_tp in names(RFILES[[val_strat]])){
      # init
      c_fn = RFILES[[val_strat]][[model_tp]]
      if(c_fn == -1)
        next
      # log
      c_nm = sprintf("%s-%s", val_strat, model_tp)
      print("*******")
      print(c_nm)
      # process
      tmp = add_to_all_preds_dfs_cur_res(c_fn, val_strat, model_tp, all_preds_df, all_final_preds_df)
      all_preds_df = tmp$all_preds_df
      all_final_preds_df = tmp$all_final_preds_df
    }
  }
  return(list(all_preds_df = all_preds_df, all_final_preds_df = all_final_preds_df))
}

# get predictions per record from all models in a long table format
compute_per_record_preds <- function(all_preds_df) {
  per_record_preds = as.data.frame(list(center = 0,
                                        outcome = 0,
                                        row_id = 0,
                                        global = 0,
                                        local = 0,
                                        fedavg_no_recal = 0,
                                        fedavg_glob_recal = 0,
                                        ensemble_no_recal = 0,
                                        ensemble_glob_recal = 0,
                                        val_strat = 0))[0,]
  for(c_row_id in sort(uniq(all_preds_df$row_id))) {
    # print(c_row_id)
    c_rows = all_preds_df[all_preds_df$row_id == c_row_id,]
    if(len(uniq(c_rows$outcome)) != 1 || len(uniq(c_rows$center)) != 1){
      print("len(uniq(c_rows$outcome)) != 1 || len(uniq(c_rows$center)) != 1")
      exit(-1)
    }
    c_rows_lcoa = c_rows[c_rows$val_strat == 'LCOA',]
    c_rows_cv = c_rows[c_rows$val_strat == 'CV',]

    n_row = as.data.frame(list(center = c_rows$center[1],
                               outcome = c_rows$outcome[1],
                               row_id = c_row_id))

    n_row_lcoa = n_row
    for(c_mds in c_rows_lcoa$MDS)
      n_row_lcoa[,c_mds] = c_rows_lcoa[c_rows_lcoa$MDS==c_mds,]$prediction
    n_row_lcoa$val_strat = 'LCOA'

    n_row_cv = n_row
    for(c_mds in c_rows_cv$MDS)
      n_row_cv[,c_mds] = c_rows_cv[c_rows_cv$MDS==c_mds,]$prediction
    n_row_cv$val_strat = 'CV'
    missing_cols = setdiff( cns(per_record_preds), cns(n_row_cv))
    n_row_cv[,missing_cols] = NA
    missing_cols = setdiff( cns(per_record_preds), cns(n_row_lcoa))
    n_row_lcoa[,missing_cols] = NA
    per_record_preds = rbind(per_record_preds, n_row_cv)
    per_record_preds = rbind(per_record_preds, n_row_lcoa)
  }
  return(per_record_preds)
}

get_preds_df_from_res_obj <- function(res_o, remove_final_model_preds = T){
  if(remove_final_model_preds)
    return(res_o$preds_df[res_o$preds_df$fold != max(res_o$preds_df$fold),])
  return(res_o$preds_df)
}

get_mean_metric_from_result_obj <- function(res_o, metric_nm = 'AUC-ROC', meta_pool_locals = F) {
  # take predictions from all folds minus the final one
  preds_df = get_preds_df_from_res_obj(res_o, remove_final_model_preds = T)
  table(preds_df$center)
  is_local_model = "local-models-only" %in% class(res_o[["fitted_models"]][[1]])
  is_cv = is_local_model || sd(table(preds_df$fold)) < N_FOLDS_CV # local models only have cv
  is_lcoa = !is_cv
  stratify_on = if(is_cv) 'fold' else 'center'
  meta_res = NULL
  if(is_lcoa)
    meta_res = try_meta_analyse_from_res_obj(res_o, metric = metric_nm, stratify_on = stratify_on)
  else
    meta_res = try_pool_CV_performance_from_res_obj(res_o, metric = metric_nm, stratify_on = stratify_on,
                                                    meta_pool_locals = meta_pool_locals)

  ret_o = list(mean_with_95ci = c(meta_res$pooled_mean_ci_low, meta_res$pooled_mean, meta_res$pooled_mean_ci_hi),
               meta_df = if(is_lcoa) meta_res$external_only else meta_res$fold_means, # todo: meta_df = bad naming...
               individual_means = if(is_lcoa) meta_res$center_means else meta_res$fold_means,
               ind_lbs = if(is_lcoa) meta_res$center_lbs else meta_res$fold_lbs,
               ind_ubs = if(is_lcoa) meta_res$center_ubs else meta_res$fold_ubs)
  return(ret_o)
}

uniq_and_group_predictors <- function( c_predictors ) {
  c_predictors = uniq(c_predictors)
  grouped_preds = c()
  for(c_var in c_predictors) {
    group_found = F
    for(var_cat in POSSIBLE_VAR_CATEGORIES) {
      if(c_var %in% var_cat){
        grouped_preds = c(grouped_preds, var_cat[1])
        group_found = T
        break
      }
    }
    if(!group_found)
      grouped_preds = c(grouped_preds, c_var)
  }
  return(uniq(grouped_preds))
}

try_meta_analyse_from_r_df <- function(r_df, metric, stratify_on = 'center', res_o = NULL) {
  c_nums = r_df[,stratify_on]
  # fix colname auc_95_hi -> auc_95hi
  colnames(r_df) = c(cns(r_df)[1:3], 'auc_95hi', cns(r_df)[5:ncol(r_df)])
  col_selector = 'auc'
  if(metric == 'calibration-intercept')
    col_selector = 'cali_int'
  if(metric == 'calibration-slope')
    col_selector = 'cali_slp'
  c_means = r_df[,col_selector]
  c_lbs = r_df[,sprintf('%s_95lo',col_selector)]
  c_ubs = r_df[,sprintf('%s_95hi',col_selector)]
  # TODO: what is the right volume (n) here????????
  c_volumes = rep(1,nrow(r_df))

  c_sds = get_sd_from_ci95(c_lbs, c_ubs, c_volumes)
  # prep df in format for claling metamean
  meta_df = as.data.frame(list(  n.e = rep(1,len(c_means)),
                                 mean.e = c_means,
                                 sd.e = c_sds,
                                 n.c = rep(-1,len(c_means)),
                                 mean.c = rep(-1,len(c_means)),
                                 sd.c = rep(-1,len(c_means)),
                                 studlab = names(inv_hidden_center_mapping )[as.numeric(c_nums)]
  ))
  xxx = meta_df
  transform_fn = identity
  rev_transform_fn = identify
  if(metric == "AUC-ROC") {
    transform_fn = gtools::logit
    rev_transform_fn = gtools::inv.logit
  }

  if(metric == "calibration-intercept" || metric == 'calibration-slope') {
    transform_fn = identity
    rev_transform_fn = identity
  }
  if(metric == 'AUC-ROC'){
    if(any(c_ubs ==1))
      c_ubs[which(c_ubs ==1)] = 0.99999 # just to get a SD that is not nan
  }
  t_c_ubs = transform_fn(c_ubs)
  t_c_lbs = transform_fn(c_lbs)
  t_sds = get_sd_from_ci95(t_c_lbs, t_c_ubs, c_volumes)
  # apply any needed transform on metric mean being meta-analysed
  xxx$mean.e = transform_fn(xxx$mean.e)
  # apply same transform on the sd of the metric
  xxx$sd.e = t_sds
  ext_meta = meta::metamean(n = n.e, mean = mean.e, sd = sd.e, studlab= studlab, data = xxx , fixed = F, random = T, prediction = T )
  # add any missed centers/folds as NAs into c_means
  if(!is.null(res_o)){
    missed_cfs = setdiff(uniq(levels(res_o$preds_df[,stratify_on])[res_o$preds_df[,stratify_on]]), c_nums)
    if(len(missed_cfs) > 0)
      for(c_miss in sort(as.numeric(missed_cfs))){
        c_means = c(c_means[1:(c_miss-1)], NA, c_means[c_miss:len(c_means)])
        c_lbs = c(c_lbs[1:(c_miss-1)], NA, c_lbs[c_miss:len(c_lbs)])
        c_ubs = c(c_ubs[1:(c_miss-1)], NA, c_ubs[c_miss:len(c_ubs)])
      }
  }

  return(list(pooled_mean = rev_transform_fn(ext_meta$TE.random),
              pooled_mean_ci_low = rev_transform_fn(ext_meta$lower.random),
              pooled_mean_ci_hi = rev_transform_fn(ext_meta$upper.random),
              external_only = ext_meta, meta_df = meta_df, center_means = c_means,
              center_lbs = c_lbs,
              center_ubs = c_ubs))
}

try_meta_analyse_from_res_obj <- function(res_o, metric, stratify_on = 'center') {
  res_o$preds_df = get_preds_df_from_res_obj(res_o, remove_final_model_preds = T)
  r_df = extract_results_df(res_o$preds_df, plot_cali_graph = F, stratify_on = stratify_on)
  r_df = r_df[!r_df[,stratify_on] %in% c('ALL', 'mean'),]
  r_df$volume = NA
  res_o$preds_df[, stratify_on] = as.numeric(levels(res_o$preds_df[, stratify_on])[res_o$preds_df[, stratify_on]])
  for(c_strat in sort(uniq(res_o$preds_df[,stratify_on]))) {
    c_vol = howmany(res_o$preds_df[, stratify_on] == c_strat)
    r_df[r_df[, stratify_on] == as.character(c_strat),]$volume = c_vol
  }
  return(try_meta_analyse_from_r_df(r_df = r_df, metric = metric, stratify_on = stratify_on, res_o = res_o))
}

try_pool_CV_performance_from_res_obj <- function(res_o, metric, stratify_on = 'center',
                                                 meta_pool_locals = F) {
  res_o$preds_df = get_preds_df_from_res_obj(res_o, remove_final_model_preds = T)

  # in case we have exact zero value as prediciton, set it to lowest nonzero pred
  # (in order to be able to compute calibration)
  zero_preds = which(res_o$preds_df$prediction == 0)
  lowest_nonzero_pred = if(len(zero_preds) != 0 ) min(res_o$preds_df$prediction[-zero_preds]) else min(res_o$preds_df$prediction)
  res_o$preds_df$prediction[zero_preds] = lowest_nonzero_pred

  one_preds = which(res_o$preds_df$prediction == 1)
  highenst_nonone_pred = if(len(one_preds) != 0 ) max(res_o$preds_df$prediction[-one_preds]) else max(res_o$preds_df$prediction)
  res_o$preds_df$prediction[one_preds] = highenst_nonone_pred

  is_local_models = res_o$preds_df$model_name[1] == 'lr-lasso-locals'

  r_df = extract_results_df(res_o$preds_df, plot_cali_graph = F,
                            stratify_on = if(is_local_models && meta_pool_locals) c(stratify_on, 'center') else stratify_on)
  # fix colname auc_95_hi -> auc_95hi
  colnames(r_df) = c(cns(r_df)[1:3], 'auc_95hi', cns(r_df)[5:ncol(r_df)])
  # do meta analysis pooling here
  if(is_local_models && meta_pool_locals) {
    meta_df = r_df[r_df$fold =='mean',]
    meta_res = try_meta_analyse_from_r_df(r_df = meta_df, metric = 'AUC-ROC', stratify_on = 'center',
                                          res_o = NULL)
    col_selector = 'auc'
    if(metric == 'calibration-intercept')
      col_selector = 'cali_int'
    if(metric == 'calibration-slope')
      col_selector = 'cali_slp'

    c_means = meta_df[,col_selector]
    c_lbs = meta_df[,sprintf('%s_95lo',col_selector)]
    c_ubs = meta_df[,sprintf('%s_95hi',col_selector)]

    c_nums= uniq(meta_df$center)
    missed_cfs = setdiff(as.numeric(levels(res_o$preds_df$center)), c_nums)
    if(len(missed_cfs) > 0)
      for(c_miss in sort(as.numeric(missed_cfs))){
        c_means = c(c_means[1:(c_miss-1)], NA, c_means[c_miss:len(c_means)])
        c_lbs = c(c_lbs[1:(c_miss-1)], NA, c_lbs[c_miss:len(c_lbs)])
        c_ubs = c(c_ubs[1:(c_miss-1)], NA, c_ubs[c_miss:len(c_ubs)])
      }

    ret_o = list(pooled_mean = meta_res$pooled_mean,
                 pooled_mean_ci_low = meta_res$pooled_mean_ci_low,
                 pooled_mean_ci_hi = meta_res$pooled_mean_ci_hi,
                 meta_df = meta_res$meta_df,
                 fold_means = c_means,
                 fold_lbs = c_lbs,
                 fold_ubs = c_ubs)
  }
  # do standard mean pooling here
  if(!is_local_models || !meta_pool_locals) {
    r_df = r_df[!r_df[,stratify_on] %in% c('ALL', 'mean'),]
    c_nums = r_df[,stratify_on]
    col_selector = 'auc'
    if(metric == 'calibration-intercept')
      col_selector = 'cali_int'
    if(metric == 'calibration-slope')
      col_selector = 'cali_slp'
    c_means = r_df[,col_selector]
    c_lbs = r_df[,sprintf('%s_95lo',col_selector)]
    c_ubs = r_df[,sprintf('%s_95hi',col_selector)]
    # add any missed centers/folds as NAs into c_means
    missed_cfs = setdiff(uniq(levels(res_o$preds_df[,stratify_on])[res_o$preds_df[,stratify_on]]), c_nums)
    if(len(missed_cfs) > 0)
      for(c_miss in sort(as.numeric(missed_cfs))){
        c_means = c(c_means[1:(c_miss-1)], NA, c_means[c_miss:len(c_means)])
        c_lbs = c(c_lbs[1:(c_miss-1)], NA, c_lbs[c_miss:len(c_lbs)])
        c_ubs = c(c_ubs[1:(c_miss-1)], NA, c_ubs[c_miss:len(c_ubs)])
      }
    mean_of_means = mean(c_means, na.rm =T)
    ci95 = get_ci_95_from_mean_and_sd(mean_of_means, sd(c_means, na.rm=T), howmany(!is.na(c_means)))
    ret_o = list(pooled_mean = mean_of_means,
                 pooled_mean_ci_low = ci95[1],
                 pooled_mean_ci_hi = ci95[2],
                 meta_df = NA, fold_means = c_means,
                 fold_lbs = c_lbs,
                 fold_ubs = c_ubs)
  }
  return(ret_o)
}

#' @title  Calculate net reclassification improvement (NRI) betweeen two sets of (paired) predictions
#' @return single numeric value of the NRI. Ranges from -2 to 2.
#' Tests if prediction from preds2 are better than those from preds1
#
# with negative values indicating worse reclassification
# and positive values indicating better reclassification by Model 2 compared to Model 1.
calculate_NRI <- function(preds1, preds2, outs, alt_formula = T, round_digits = -1, verbose = F) {
  # Initialize counts
  correct_up <- 0
  correct_down <- 0
  incorrect_up <- 0
  incorrect_down <- 0
  # make any missing predictions equal the mean prevalance of the outcome
  # m_prev = howmany(outs == 1) / len(outs)
  # preds2 = umap(preds2, function(x) {if(is.na(x)) m_prev else x})
  # preds1 = umap(preds1, function(x) {if(is.na(x)) m_prev else x})
  if(round_digits > 0) {
    preds1 = try_round(preds1, round_digits)
    preds2 = try_round(preds2, round_digits)
  }

  #remove records missing both predictions
  missing_preds_idxs = union(which(is.na(preds1)), which(is.na(preds2)))
  if(len(missing_preds_idxs) > 0) {
    preds1 = preds1[-missing_preds_idxs]
    preds2 = preds2[-missing_preds_idxs]
  }

  # Iterate over indices
  # if equal give half
  for (i in seq_along(preds2)) {
    if (preds1[i] > preds2[i] && outs[i] == 1) { # did preds1 give a higher risk when outcome was positive?
      correct_up <- correct_up + 1
    } else if (preds1[i] < preds2[i] && outs[i] == 1) { # did preds1 give a lower risk when outcome was positive?
      incorrect_up <- incorrect_up + 1
    } else if (preds1[i] < preds2[i] && outs[i] == 0) { # did preds1 give a lower risk when outcome was negative?
      correct_down <- correct_down + 1
    } else if (preds1[i] > preds2[i] && outs[i] == 0) { # did preds1 give a higher risk when outcome was negative?
      incorrect_down <- incorrect_down + 1
    } else if (preds1[i] == preds2[i]) { # did preds2 == preds1? *debatable, could also add nothing
      correct_up <- correct_up + 0.5
      correct_down <- correct_down + 0.5
      incorrect_down <- incorrect_down + 0.5
      incorrect_up <- incorrect_up + 0.5
    }
  }

  # Calculate NRI
  n <- len(preds2)
  nevents = howmany(outs ==1)
  nnoevents = howmany(outs ==0)
  if(nnoevents == 0) # dont divide by zero
    nnoevents = 1
  if(nevents == 0)
    nevents = 1
  # alternative formula :
  NRI_alternative = ((correct_up - incorrect_up ) / nevents) + ((correct_down - incorrect_down ) / nnoevents)
  NRI = (correct_up + correct_down) / n
  if(verbose)
    try_log_debug("\tNRI_alt = %0.2f;\t NRI = %0.2f;", NRI_alternative, NRI)
  if(alt_formula)
    return(NRI_alternative)
  return(NRI)
}

#' @title Compute NRI for list of model predicitons
#' @param preds_pr - dataframe  with predictions per record (row) and model (column)
#' @clms - column names of model predictions from preds_pr to use
compute_nri_matrix <- function(preds_pr, clms, bootstrap_n = 1, round_digits = -1, alt_formula=T, verbose = 0) {
  # init nri_matrix
  nri_matrix = list()
  clms_iter = c()
  n_rec = nrow(preds_pr)
  if(bootstrap_n > 1) {
    for(c_clm in clms){
      clms_iter = c(clms_iter, sprintf("%s_lb",c_clm), c_clm, sprintf("%s_ub",c_clm))
    }
  }
  else
    clms_iter = clms

  for(c_clm in clms_iter){
    nri_matrix[[c_clm]] = rep(NA, len(clms_iter))
  }
  nri_matrix = as.data.frame(nri_matrix)
  rownames(nri_matrix) = cns(nri_matrix)
  # decide to keep lb/ub only on rows, no need to repeat that much
  nri_matrix = nri_matrix[,-grep("_lb",cns(nri_matrix))]
  nri_matrix = nri_matrix[,-grep("_ub",cns(nri_matrix))]
  # View(nri_matrix)
  for(m1_idx in 1:(len(clms)-1) ){
    m1 = clms[m1_idx]
    for(m2_idx in (m1_idx+1):len(clms) ){
      m2 = clms[m2_idx]
      c_preds1 =  preds_pr[,m1]
      c_preds2 = preds_pr[,m2]
      if(bootstrap_n > 1) {
        nri_smpls = umap(1:bootstrap_n, function(x){
          sample_idxs = sample(1:n_rec, replace = T)
          calculate_NRI(c_preds1[sample_idxs], c_preds2[sample_idxs],
                        preds_pr$outcome[sample_idxs], alt_formula = alt_formula, round_digits = round_digits,
                        verbose = F)
        })

        c_nri = mean(nri_smpls)

        c_nri_lb = quantile(nri_smpls, probs = c(0.025))
        c_nri_ub = quantile(nri_smpls, probs = c(0.975))

        nri_matrix[m1,m2] = c_nri
        nri_matrix[m2,m1] = if(alt_formula) -c_nri else 1-c_nri
        # decide to keep lb/ub only on rows, no need to repeat that much
        nri_matrix[sprintf("%s_lb",m1), m2] = c_nri_lb
        nri_matrix[sprintf("%s_ub",m1), m2] = c_nri_ub

        nri_matrix[sprintf("%s_lb",m2), m1] = if(alt_formula) -c_nri_ub else 1-c_nri_ub
        nri_matrix[sprintf("%s_ub",m2), m1] = if(alt_formula) -c_nri_lb else 1-c_nri_lb

        if(verbose > 0) {
          if(c_nri_lb > 0)
            try_log_debug(sprintf("%s >> %s (%0.2f/%0.2f)", m1, m2, c_nri_lb, c_nri_ub))
          if(c_nri_ub < 0)
            try_log_debug(sprintf("%s >> %s (%0.2f/%0.2f)", m2, m1, if(alt_formula) -c_nri_ub else 1-c_nri_ub,
                                  if(alt_formula) -c_nri_lb else 1-c_nri_lb))
        }
        if(verbose > 1) {
          try_log_debug(sprintf("%s == %s (%0.2f/%0.2f)", m1, m2, c_nri_lb, c_nri_ub))
          try_log_debug(sprintf("%s == %s (%0.2f/%0.2f)", m2, m1, -c_nri_ub, if(alt_formula) -c_nri_lb else 1-c_nri_lb))
        }

      }
      else {
        c_nri = calculate_NRI(c_preds1, c_preds2, preds_pr$outcome, verbose = verbose > 1,
                              alt_formula = alt_formula, round_digits = round_digits)
        nri_matrix[m1,m2] = c_nri
        nri_matrix[m2,m1] = -c_nri
        if(verbose > 0)
          try_log_debug(sprintf("%s >> %s", if( (alt_formula && c_nri>0) || (!alt_formula && c_nri>0.5) ) m1 else m2,
                                if( (alt_formula && c_nri>0) || (!alt_formula && c_nri>0.5) ) m2 else m1 ))
      }
    }
  }
  return(nri_matrix)
}


###### plot functions ######

#' plot calibration graphs for multiple models
#'
#' @param strat_on - stratify on either center or fold. Used for computing mean calibration metric values in legend
#' @param skip_MDSs - list of model development strategies (MDSs) to skip. Default = skip none.
#' @param legend_contents - list of zero or more length containing either: (recomputed/given)_(int/slp)_(mean/95ci)
#' if recomputed legend content, uses only predictions outside of excluded_percentiles to calculate.
plot_cali_overlay <- function(c_df, df_metrics, val_strat = "CV", excluded_percentiles = c(0,100), xlim = c(0,1), ylim =c(0,1),
                              legend_correction = c(0.28, 1.0), df = 6, first_title = "Calibration graph for" ,
                              extra_title_txt = "", scale_preds = F,
                              strat_on = 'fold',
                              legend_contents = c("recomputed_int_95ci", "recomputed_slp_95ci"),
                              skip_MDSs = c(),
                              mds_to_lty_mapping = NULL ) {
  is_LCOA = val_strat == "LCOA"
  is_CV = val_strat == "CV"
  if(!is_LCOA && !is_CV) {
    try_log_error("Uknown val_strat. Must be one of: (CV, LCOA)")
    return()
  }
  val_strat_title = if(is_LCOA) "Leave-Center-Out Analysis" else "Cross-Validation"
  if(first_title == "") val_strat_title = ""
  if(is_LCOA) {
    colors = RColorBrewer::brewer.pal(5, "Set1")[-1][c(1,4,2)]
    colors_light = RColorBrewer::brewer.pal(5, "Pastel1")[-1][c(1,4,2)]

  }
  if(is_CV) {
    colors = RColorBrewer::brewer.pal(5, "Set1")[-1][c(1,3,4,2)]
    colors_light = RColorBrewer::brewer.pal(5, "Pastel1")[-1][c(1,3,4,2)]
  }

  i = 1
  legend_txt = c()
  ltys = c()
  for(mds in uniq(c_df$MDS)) {  # cv order   [c(1,2,4,3)] , lcoa [c(1,3,2)]
    if(mds %in% skip_MDSs)
      next
    c_preds = c_df[c_df$MDS == mds,]
    for(c_strat in uniq(c_df[,strat_on])){
      c_strat_preds = c_preds[c_preds[,strat_on] == c_strat, ]$prediction
      c_strat_preds = if(scale_preds) feature_scale(c_strat_preds) else c_strat_preds
      c_preds[c_preds[,strat_on] == c_strat, ]$prediction =  c_strat_preds
    }
    c_outs = c_preds$outcome
    c_preds = c_preds$prediction

    # c_preds = c_df[c_df$MDS == mds,]$prediction
    # c_outs = c_df[c_df$MDS == mds,]$outcome

    c_mds = mds
    # using current cali instead of pooled one in per-center plots
    c_pooled_cali = df_metrics[df_metrics$model_nm == mds,]
    c_pooled_cali = c_pooled_cali[c_pooled_cali$val_strat == val_strat,]
    c_pooled_cali_int = c_pooled_cali[,c('cali_int_95lo', 'cali_int', 'cali_int_95hi')]
    c_pooled_cali_slp = c_pooled_cali[,c('cali_slp_95lo', 'cali_slp', 'cali_slp_95hi')]
    if(c_mds == "global")
      c_mds = "Central"
    if(c_mds == "ensemble_glob_recal" || c_mds == "ensemble_no_recal")
      c_mds = "Ensemble"
    if(c_mds == "local")
      c_mds = "Local"
    if(c_mds == "fedavg_no_recal" || c_mds == "fedavg_glob_recal")
      c_mds = "FedAvg"
    c_lty = 1
    if(!is.null(mds_to_lty_mapping) && !is.null(mds_to_lty_mapping[[c_mds]]))
      c_lty = mds_to_lty_mapping[[c_mds]]
    ltys = c(ltys, c_lty)


    excl_percentile_lower = excluded_percentiles[1]
    excl_percentile_upper = excluded_percentiles[2]

    cs_preds = c_preds
    cs_out = c_outs
    cc_df = c_df[c_df$MDS == mds,]
    cal_df = NULL
    for(c_strat in  uniq(c_df[,strat_on])){
      c_idxs = which(cc_df[,strat_on] == c_strat)
      cc_preds = cs_preds[c_idxs]
      cc_outs = cs_out[c_idxs]

      lowr_bound = quantile(cc_preds, probs = excl_percentile_lower/100)
      upr_bound = quantile(cc_preds, probs = excl_percentile_upper/100)

      selected_idxs =  intersect(which(cc_preds >= lowr_bound), which(cc_preds <= upr_bound))
      cc_preds = cc_preds[selected_idxs]
      cc_outs = cc_outs[selected_idxs]
      cc_cali = try_get_calibration(cc_preds, cc_outs, verbose = F)

      intmean = try_round(cc_cali$cal.intercept,2)
      intcihi = try_round(cc_cali$ci.cal.intercept[2],2)
      intcilo = try_round(cc_cali$ci.cal.intercept[1],2)
      intsd = (cc_cali$ci.cal.intercept[2] - cc_cali$ci.cal.intercept[1]) / (2*1.96*sqrt(len(selected_idxs)))

      slpmean = try_round(cc_cali$cal.slope,2)
      slpcihi = try_round(cc_cali$ci.cal.slope[2],2)
      slpcilo = try_round(cc_cali$ci.cal.slope[1],2)
      slpsd = (cc_cali$ci.cal.slope[2] - cc_cali$ci.cal.slope[1]) / (2*1.96*sqrt(len(selected_idxs)))


      nvals_list = list(intmean = intmean, intcihi = intcihi, intcilo = intcilo, intsd = intsd,
                        slpmean = slpmean, slpcihi = slpcihi, slpcilo = slpcilo, slpsd = slpsd )
      nvals_list[[strat_on]] = c_strat
      nvals_list$volume = len(selected_idxs)
      cal_df = rbind(cal_df, as.data.frame(nvals_list))
    }
    pooled_means = list(intmean = NA, intcilo = NA, intcihi = NA, intsd = NA,
                        slpmean = NA, slpcilo = NA, slpcihi = NA, slpsd = NA  )

    if(val_strat == 'CV'){
      pooled_means$intmean = mean(cal_df$intmean, na.rm = T)
      n_nn = howmany(!is.na(cal_df$intmean))
      c_sd = sd(cal_df$intmean, na.rm = T)
      c_se = c_sd /sqrt(n_nn)
      moe = c_se * 1.96
      pooled_means$intcilo = pooled_means$intmean - moe
      pooled_means$intcihi = pooled_means$intmean + moe
      pooled_means$intsd = c_sd

      pooled_means$slpmean = mean(cal_df$slpmean, na.rm = T)
      n_nn = howmany(!is.na(cal_df$slpmean))
      c_sd =  sd(cal_df$slpmean, na.rm = T)
      c_se = c_sd /sqrt(n_nn)
      moe = c_se * 1.96
      pooled_means$slpcilo = pooled_means$slpmean - moe
      pooled_means$slpcihi = pooled_means$slpmean + moe
      pooled_means$slpsd = c_sd

      pooled_means = try_round(as.data.frame(pooled_means),4)
    }
    if(val_strat == 'LCOA'){
      #c int
      c_means = cal_df$intmean
      c_sds = cal_df$intsd
      c_nums = cal_df[, strat_on]
      xxx = as.data.frame(list(  n.e = rep(1,len(c_means)),
                                 mean.e = c_means,
                                 sd.e = c_sds,
                                 n.c = rep(-1,len(c_means)),
                                 mean.c = rep(-1,len(c_means)),
                                 sd.c = rep(-1,len(c_means)),
                                 studlab = names(inv_hidden_center_mapping )[as.numeric(c_nums)]
      ))

      meta_res = meta::metamean(n = n.e, mean = mean.e, sd = sd.e, studlab= studlab, data = xxx , fixed = F, random = T, prediction = T )
      pooled_means$intsd = try_round(meta_res$seTE.random*sqrt(sum(meta_res$n)),4)
      meta_int = as.data.frame(list(pooled_mean = (meta_res$TE.random),
                                    pooled_mean_ci_low = (meta_res$lower.random),
                                    pooled_mean_ci_hi = (meta_res$upper.random)))

      #c slp
      c_means = cal_df$slpmean
      c_sds = cal_df$slpsd
      c_nums = cal_df[, strat_on]
      xxx = as.data.frame(list(  n.e = rep(1,len(c_means)),
                                 mean.e = c_means,
                                 sd.e = c_sds,
                                 n.c = rep(-1,len(c_means)),
                                 mean.c = rep(-1,len(c_means)),
                                 sd.c = rep(-1,len(c_means)),
                                 studlab = names(inv_hidden_center_mapping )[as.numeric(c_nums)]
      ))
      meta_res = meta::metamean(n = n.e, mean = mean.e, sd = sd.e, studlab= studlab, data = xxx , fixed = F, random = T, prediction = T )
      pooled_means$slpsd = try_round(meta_res$seTE.random*sqrt(sum(meta_res$n)),4)
      meta_slp = as.data.frame(list(pooled_mean = (meta_res$TE.random),
                                    pooled_mean_ci_low = (meta_res$lower.random),
                                    pooled_mean_ci_hi = (meta_res$upper.random)))

      pooled_means$intmean = try_round(meta_int$pooled_mean, 4)
      pooled_means$intcilo = try_round(meta_int$pooled_mean_ci_low, 4)
      pooled_means$intcihi = try_round(meta_int$pooled_mean_ci_hi, 4)

      pooled_means$slpmean = try_round(meta_slp$pooled_mean, 4)
      pooled_means$slpcilo = try_round(meta_slp$pooled_mean_ci_low, 4)
      pooled_means$slpcihi = try_round(meta_slp$pooled_mean_ci_hi, 4)
      pooled_means = as.data.frame(pooled_means)
    }
    pooled_means
    pooled_means = try_round(pooled_means,2)

    intmean = pooled_means$intmean
    intcilo = pooled_means$intcilo
    intcihi = pooled_means$intcihi
    intsd = pooled_means$intsd
    slpmean = pooled_means$slpmean
    slpcilo = pooled_means$slpcilo
    slpcihi = pooled_means$slpcihi
    slpsd = pooled_means$slpsd
    # todo: if we are overriding int/slp vars here can alos simplify logic for legend contents...
    if(len(grep("given", legend_contents[1])) !=0 ) #  heuristic - infer if all are given or computed , todo: make it safer!
    {
      intmean = c_pooled_cali_int[2]
      intcilo = c_pooled_cali_int[1]
      intcihi = c_pooled_cali_int[3]
      # intsd = pooled_means$intsd
      slpmean = c_pooled_cali_slp[2]
      slpcilo = c_pooled_cali_slp[1]
      slpcihi = c_pooled_cali_slp[3]
      # slpsd = pooled_means$slpsd
    }

    cox_intercept_ok = intcilo <= 0 && intcihi >= 0
    if(len(cox_intercept_ok) == 0 || is.na(cox_intercept_ok))
      cox_intercept_ok = F
    cox_slope_ok = slpcilo <= 1 && slpcihi >= 1
    if(len(cox_slope_ok) == 0 || is.na(cox_slope_ok))
      cox_slope_ok = F

    cox_calibrate_ok = cox_intercept_ok && cox_slope_ok
    cox_miscali_txt = if(cox_calibrate_ok) "" else if(cox_intercept_ok) "^" else "*"

    l_contents = ""
    # ist of zero or more length containing either: (recomputed/given)_(int/slp)_(mean/95ci)
    for(l_c in legend_contents) {
      is_last = which(legend_contents == l_c) == len(legend_contents)
      is_first = strlen(l_contents) == 0
      c_val_str = NULL
      if(l_c == "recomputed_int_95ci")
        c_val_str = sprintf("[%0.2f/%0.2f]", intcilo, intcihi)
      if(l_c == "recomputed_slp_95ci")
        c_val_str = sprintf("[%0.2f/%0.2f]", slpcilo, slpcihi)
      if(l_c == "recomputed_int_mean") {
        float_nm = sprintf("%%0.%df", max(2,count_decimal_digits(intmean)))
        c_val_str = sprintf(float_nm, intmean)
      }
      if(l_c == "recomputed_slp_mean") {
        float_nm = sprintf("%%0.%df", max(2,count_decimal_digits(slpmean)))
        c_val_str = sprintf(float_nm, slpmean)
      }

      if(l_c == "given_int_95ci")
        c_val_str = sprintf("[%0.2f/%0.2f]", c_pooled_cali_int[1], c_pooled_cali_int[3])
      if(l_c == "given_slp_95ci")
        c_val_str = sprintf("[%0.2f/%0.2f]", c_pooled_cali_slp[1], c_pooled_cali_slp[3])
      if(l_c == "given_int_mean") {
        float_nm = sprintf("%%0.%df", max(2,count_decimal_digits(c_pooled_cali_int[2])))
        c_val_str = sprintf(float_nm, c_pooled_cali_int[2])
      }
      if(l_c == "given_slp_mean") {
        float_nm = sprintf("%%0.%df", max(2,count_decimal_digits(c_pooled_cali_slp[2])))
        c_val_str = sprintf(float_nm, c_pooled_cali_slp[2])
      }



      if(is_first)
        l_contents = sprintf("%s", c_val_str)
      if(!is_first && is_last)
          l_contents = sprintf("%s; %s", l_contents, c_val_str)

      if(!is_last && !is_first)
          l_contents = sprintf("%s %s; ", l_contents, c_val_str)

    }
    c_txt = sprintf("%s%s (%s)", c_mds, cox_miscali_txt, l_contents)
    legend_txt = c(legend_txt, c_txt)
    percentiles_text = if(len(excluded_percentiles) == 2 &&
                          (excluded_percentiles[1] != 0 || excluded_percentiles[2] != 100))
      sprintf("\n(top %0.1f%% predicted probabilities excluded)", 100-excluded_percentiles[2]) else ""


    plot_calibration_graph(c_preds, c_outs,
                           title = sprintf("%s %s%s%s", first_title, val_strat_title, extra_title_txt, percentiles_text),
                           smoothed = F, line_par = list(col = colors[i], lwd = 5, lty = c_lty),
                           add_quantiles_and_mean_text = F, replace = i==1, shade_col = colors_light[i], shade_density = 0,
                           xlim = xlim, y_lim= ylim,
                           excluded_percentiles = excluded_percentiles,
                           df = df, with_quantile_bars = F)
    i = i + 1
  }
  legend( legend_correction[1], legend_correction[2], legend_txt, col = colors, lwd = 6, lty = ltys)
  return()
}

#' plot AUC-ROC curves for multiple models
#'
#' @param skip_MDSs - list of model development strategies (MDSs) to skip. Default = skip none.
plot_aucs_overlay <- function(c_df, df_metrics, title = "", val_strat = "CV" , use_pooled_metric = F,
                              legend_xy = c(0.64, 0.24), skip_MDSs = c(), legend_include_sd = T,
                              mds_to_lty_mapping = NULL) {
  is_LCOA = val_strat == "LCOA"
  is_CV = val_strat == "CV"
  if(!is_LCOA && !is_CV) {
    try_log_error("Uknown val_strat. Must be one of: (CV, LCOA)")
    return()
  }

  c_aucs = list()
  c_volumes = list()
  strat_on = if(is_CV) "fold" else "center"

  for(mds in uniq(c_df$MDS)) {
    if(mds %in% skip_MDSs)
      next

    c_preds = c_df[c_df$MDS == mds,]
    for(c_strat in uniq(c_df[,strat_on])){
      c_strat_preds = c_preds[c_preds[,strat_on] == c_strat, ]$prediction
      c_strat_preds = feature_scale(c_strat_preds)
      c_preds[c_preds[,strat_on] == c_strat, ]$prediction =  c_strat_preds
    }
    c_outs = c_preds$outcome
    c_volumes[[len(c_volumes)+1]] = len(c_outs)
    c_preds = c_preds$prediction
    c_auc = roc(c_outs, c_preds)
    c_aucs[[len(c_aucs)+1]] = list(auc=c_auc, mds = mds)
  }
  colors = RColorBrewer::brewer.pal(5, "Set1")[-1]
  # 1 = central # 2 = fedavg # 3 = ensemble [LCOA]
  colors_const = c("#377EB8", "#4DAF4A", "#984EA3", "#FF7F00")
  if(is_LCOA) {
    colors = colors[c(1,4,2)]
  }

  if(is_CV) {
    is_missing_local = len(c_aucs) == 3
    colors = colors[c(1,3,4,2)]
    if(is_missing_local) {
      c_aucs =list(c_aucs[[1]], c_aucs[[1]], c_aucs[[2]], c_aucs[[3]])
      c_aucs[[2]] = list()
    }
    else
      c_aucs = list(c_aucs[[1]], c_aucs[[2]], c_aucs[[3]], c_aucs[[4]])
  }
  mdss = c()
  ltys = c()
  for(i in 1:len(c_aucs)){
    if(len(c_aucs[[i]]) == 0)
      next
    c_roc = c_aucs[[i]]$auc
    c_mds = c_aucs[[i]]$mds
    c_volume = c_volumes[[i]]
    c_pooled_auc = df_metrics[df_metrics$model_nm == c_mds,]
    c_pooled_auc = c_pooled_auc[c_pooled_auc$val_strat == val_strat,]
    c_pooled_auc = c_pooled_auc[,c('auc_95lo', 'auc', 'auc_95hi')]
    if(c_mds == "global")
      c_mds = "Central"
    if(c_mds == "ensemble_glob_recal" || c_mds == "ensemble_no_recal")
      c_mds = "Ensemble"
    if(c_mds == "local")
      c_mds = "Local"
    if(c_mds == "fedavg_no_recal" || c_mds == "fedavg_glob_recal")
      c_mds = "FedAvg"
    c_lty = 1
    if(!is.null(mds_to_lty_mapping) && !is.null(mds_to_lty_mapping[[c_mds]]))
      c_lty = mds_to_lty_mapping[[c_mds]]
    ltys = c(ltys,c_lty)
    txt_auc = if(use_pooled_metric) c_pooled_auc[2] else c_roc$auc[1]
    se_auc = (c_pooled_auc[3] - c_pooled_auc[1]) /  (1.96*2)
    if(legend_include_sd)
      mdss = c(mdss, sprintf("%s (%0.2f, %0.2f)",c_mds, txt_auc, se_auc))
    else
      mdss = c(mdss, sprintf("%s (%0.2f)",c_mds, txt_auc))
    roc_plt <- plot(c_roc, print.auc = F, col = colors[i], add = i != 1, main = title , lwd = 3,
                    grid=c(0.1, 0.1), grid.col=c("grey", "grey"), lty = c_lty)
  }

  legend( legend_xy[1], legend_xy[2], mdss, col = colors, lwd = 6, lty = ltys)
}

###### init dfs #####
if(RUN_INIT_DFS) {
  # all preds df
  cat("^^^^^^^^^^^^^00000000000000000000000 all_preds_df + all_final_preds_df ^^^^^^^^^^^^^00000000000000000000000\n")
  if(1 == 1) {
    all_preds_df = NULL
    all_final_preds_df = NULL

    tmp = add_all_preds_from_RFILES_list(RFILES)
    all_preds_df = tmp$all_preds_df
    all_final_preds_df = tmp$all_final_preds_df
    rm(tmp)

  }
  cat("^^^^^^^^^^^^^00000000000000000000000 per_record_preds + final_per_record_preds ^^^^^^^^^^^^^00000000000000000000000\n")
  if(RUN_INIT_per_record_preds) {

    per_record_preds = NULL
    final_per_record_preds = NULL

    per_records_preds_sa1 = NULL
    final_per_record_preds_sa1 = NULL

    per_record_preds = compute_per_record_preds(all_preds_df)
    final_per_record_preds = compute_per_record_preds(all_final_preds_df)

  }
  cat("^^^^^^^^^^^^^00000000000000000000000 missing_local_preds ++ cv_preds_pr/lcoa_preds_pr  ^^^^^^^^^^^^^00000000000000000000000\n")
  if(1==1) {
    # looking at models fitted during CV
    cv_preds_pr = per_record_preds[per_record_preds$val_strat =='CV',]
    lcoa_preds_pr = per_record_preds[per_record_preds$val_strat =='LCOA',]

    missing_local_preds = cv_preds_pr[is.na(cv_preds_pr$local),]
    percent_missing_local_preds = try_round( (table(missing_local_preds$center)/ table(cv_preds_pr$center) *100 ) )
    count_missing_local_preds = table(missing_local_preds$center)

    mean_percent_missing_local_preds = try_round((sum(count_missing_local_preds) / sum(table(cv_preds_pr$center)) ) * 100)
    mean_count_missing_local_preds = sum(count_missing_local_preds)

    # looking at models fitted during final model
    final_cv_preds_pr = final_per_record_preds[final_per_record_preds$val_strat =='CV',]

    missing_local_preds = final_cv_preds_pr[is.na(final_cv_preds_pr$local),]
    percent_missing_local_preds = try_round( (table(missing_local_preds$center)/ table(final_cv_preds_pr$center) *100 ) )
    count_missing_local_preds = table(missing_local_preds$center)
    mean_percent_missing_local_preds = try_round((sum(count_missing_local_preds) / sum(table(final_cv_preds_pr$center)) ) * 100)
    mean_count_missing_local_preds = sum(count_missing_local_preds)

    lcoa_preds_pr = per_record_preds[per_record_preds$val_strat =='LCOA',]

  }
  cat("^^^^^^^^^^^^^00000000000000000000000 nri_matrix_list ^^^^^^^^^^^^^00000000000000000000000\n")
  nri_matrix_list = list()
  if(RUN_INIT_NRI_MATRIX) {
    # compute NRIs matrix cv
    cv_preds_pr_cpy = cv_preds_pr
    lcoa_preds_pr_cpy = lcoa_preds_pr
    # set missings to corresponding mean prevalance
    # use if you want cases where model could not be fit to get represented as a 'dumb' model that always predicts the mean outcome prevalence
    set_missing_preds_to_center_relative_prevalence <- function(all_preds_df, preds_per_record_df){
      for(c_mds in uniq(all_preds_df$MDS)){
        c_preds = preds_per_record_df[,c_mds]
        strat_on1 = "center"
        na_pred_idxs = which(is.na(c_preds))
        for(c_strat in uniq(preds_per_record_df[,strat_on1]))   {
          c_idxs = which(preds_per_record_df[,strat_on1] == c_strat)
          c_prev = howmany(preds_per_record_df[c_idxs,]$outcome == 1) / len(c_idxs)
          c_na_idxs = intersect(na_pred_idxs, c_idxs)
          if(len(c_na_idxs) > 0)
            preds_per_record_df[c_na_idxs, c_mds] = c_prev
        }
      }
      return(preds_per_record_df)
    }

    cv_preds_pr = set_missing_preds_to_center_relative_prevalence(all_preds_cv, cv_preds_pr)
    lcoa_preds_pr = set_missing_preds_to_center_relative_prevalence(all_preds_lcoa, lcoa_preds_pr)

    try_log_debug("CV NRI::")
    clms = c("global", "fedavg_no_recal", "fedavg_glob_recal", "ensemble_no_recal", "ensemble_glob_recal", "local")
    event_idxs = which(cv_preds_pr$outcome == 1)
    noevent_idxs = which(cv_preds_pr$outcome == 0)
    nri_matrix_list$CV = list()
    nri_matrix_list$CV$event = compute_nri_matrix(cv_preds_pr[event_idxs,], clms, round_digits = 2, verbose = 1, bootstrap_n = 500)
    nri_matrix_list$CV$noevent = compute_nri_matrix(cv_preds_pr[noevent_idxs,], clms, round_digits = 2, verbose = 1, bootstrap_n = 500)
    View(nri_matrix_list$CV$event[-grep("_.b",rownames(nri_matrix_list$CV$event)),])
    try_round(nri_matrix_list$CV$noevent[-grep("_.b",rownames(nri_matrix_list$CV$noevent)),], 2)

    nri_dag_ordering(nri_matrix_list$CV$event, ignore_non_sig = F)
    mtext("EVENT", side = 1)

    nri_dag_ordering(nri_matrix_list$CV$noevent, ignore_non_sig = F)
    mtext("NO EVENT", side = 1)

    try_log_debug("LCOA NRI::")
    clms = c("global", "fedavg_no_recal", "fedavg_glob_recal", "ensemble_no_recal", "ensemble_glob_recal")
    event_idxs = which(lcoa_preds_pr$outcome == 1)
    noevent_idxs = which(lcoa_preds_pr$outcome == 0)
    nri_matrix_list$LCOA = list()
    nri_matrix_list$LCOA$event = compute_nri_matrix(lcoa_preds_pr[event_idxs,], clms, round_digits = 2, verbose = 1, bootstrap_n = 500)
    nri_matrix_list$LCOA$noevent = compute_nri_matrix(lcoa_preds_pr[noevent_idxs,], clms, round_digits = 2, verbose = 1, bootstrap_n = 500)
    View(nri_matrix_list$LCOA$event[-grep("_.b",rownames(nri_matrix_list$LCOA$event)),])

    nri_dag_ordering(nri_matrix_list$LCOA$event, ignore_non_sig = F)
    mtext("EVENT", side = 1)

    nri_dag_ordering(nri_matrix_list$LCOA$noevent, ignore_non_sig = F)
    mtext("NO EVENT", side = 1)

    cv_preds_pr = cv_preds_pr_cpy
    lcoa_preds_pr = lcoa_preds_pr_cpy
  }

  cat("^^^^^^^^^^^^^00000000000000000000000 sorted_mean_reses + per_center_metrics ^^^^^^^^^^^^^00000000000000000000000\n")
  # rema mean reses
  if(RUN_INIT_POOLED_RES){
    mean_reses = NULL
    per_center_metrics = NULL

    for(val_strat in names(RFILES)) {
      for(model_tp in names(RFILES[[val_strat]])){
        c_fn = RFILES[[val_strat]][[model_tp]]
        try_log_info("[mean_reses] Loading %s %s (%s)", val_strat, model_tp, c_fn)
        if(c_fn == -1)
          next
        c_o = load_with_assign(c_fn)
        for(metric_nm in c('AUC-ROC', 'calibration-intercept', 'calibration-slope')) {
          metric_res = get_mean_metric_from_result_obj(c_o, metric = metric_nm)
          for(im_idx in 1:len(metric_res$individual_means)) {
            c_mean = metric_res$individual_means[im_idx]
            c_lb = metric_res$ind_lbs[im_idx]
            c_ub = metric_res$ind_ubs[im_idx]
            c_row = as.data.frame(list(
              val_strat = val_strat,
              model_tp = model_tp,
              metric_nm = metric_nm,
              f_c_idx = im_idx,
              metric_val = c_mean,
              metric_lb = c_lb,
              metric_ub = c_ub
            ))
            per_center_metrics = rbind(per_center_metrics, c_row)
          }

          mr_m95ci = metric_res$mean_with_95ci
          pooling_method = if(val_strat == 'LCOA') 'meta' else 'mean'
          mean_reses = rbind(mean_reses, as.data.frame(list(
            val_strat = val_strat,
            model_tp = model_tp,
            metric = metric_nm,
            mean = mr_m95ci[2],
            ci95.lower = mr_m95ci[1],
            ci95.upper = mr_m95ci[3],
            method = pooling_method,
            i2 = if(pooling_method == 'meta') metric_res$meta_df$I2 else NA,
            i2.lower = if(pooling_method == 'meta') metric_res$meta_df$lower.I2 else NA,
            i2.upper = if(pooling_method == 'meta') metric_res$meta_df$upper.I2 else NA
          )))


          cat(sprintf("%s [%s] %0.2f (%0.2f; %0.2f)\n", metric_nm, pooling_method, mr_m95ci[2], mr_m95ci[1], mr_m95ci[3]))
        }
      }
    }
    # do something special for local models
    # you have 10 folds, each has some results per center
    # for each center - compute mean metrics .
    local_preds = get_preds_df_from_res_obj(load_with_assign(RFILES$CV$local))
    local_preds$center = as.numeric(levels(local_preds$center)[local_preds$center])
    for( c_cent in sort(uniq(local_preds$center)) ) {
      c_df = local_preds[local_preds$center == c_cent,]
      c_preds = c_df$prediction
      c_outs = c_df$outcome

      # auc
      c_auc = try_get_aucroc_ci(c_preds, c_outs)
      c_row = as.data.frame(list(
        val_strat = 'CV',
        model_tp = 'local',
        metric_nm = 'AUC-ROC',
        f_c_idx = c_cent+100,   # +100 so it does not overlap with the fold ids
        metric_val = c_auc[2],
        metric_lb = c_auc[1],
        metric_ub = c_auc[3]
      ))
      per_center_metrics = rbind(per_center_metrics, c_row)

      # cali
      c_cali = try_get_calibration(c_preds, c_outs)
      c_row = as.data.frame(list(
        val_strat = 'CV',
        model_tp = 'local',
        metric_nm = 'calibration-intercept',
        f_c_idx = c_cent+100,
        metric_val = c_cali$cal.intercept,
        metric_lb = c_cali$ci.cal.intercept[1],
        metric_ub = c_cali$ci.cal.intercept[2]
      ))
      per_center_metrics = rbind(per_center_metrics, c_row)
      c_row = as.data.frame(list(
        val_strat = 'CV',
        model_tp = 'local',
        metric_nm = 'calibration-slope',
        f_c_idx = c_cent+100,
        metric_val = c_cali$cal.slope,
        metric_lb = c_cali$ci.cal.slope[1],
        metric_ub = c_cali$ci.cal.slope[2]
      ))
      per_center_metrics = rbind(per_center_metrics, c_row)
    }
    # table(local_preds$fold)

    sorted_mean_reses <- mean_reses[order(mean_reses$val_strat, mean_reses$metric, -mean_reses$mean), ]

    per_center_metrics_lcoa = per_center_metrics[per_center_metrics$val_strat=='LCOA',]

    per_center_metrics_lcoa_auc = per_center_metrics_lcoa[per_center_metrics_lcoa$metric_nm=='AUC-ROC',]

    per_center_metrics_lcoa_auc_central = per_center_metrics_lcoa_auc[per_center_metrics_lcoa_auc$model_tp == 'global',]

    per_center_metrics_lcoa_auc_fa_nr = per_center_metrics_lcoa_auc[per_center_metrics_lcoa_auc$model_tp == 'fedavg_no_recal',]
    per_center_metrics_lcoa_auc_fa_yr = per_center_metrics_lcoa_auc[per_center_metrics_lcoa_auc$model_tp == 'fedavg_glob_recal',]
    per_center_metrics_lcoa_auc_se_nr = per_center_metrics_lcoa_auc[per_center_metrics_lcoa_auc$model_tp == 'ensemble_no_recal',]
    per_center_metrics_lcoa_auc_se_yr = per_center_metrics_lcoa_auc[per_center_metrics_lcoa_auc$model_tp == 'ensemble_glob_recal',]

    try_round(quantile(per_center_metrics_lcoa_auc_central$metric_val, probs = c(0.25,0.75)), 2)
    final_vars_df_local = vars_df[vars_df$model_type == 'local',]
    final_vars_df_local = final_vars_df_local[final_vars_df_local$fold==11,]
    sort(table(final_vars_df_local$variable))

  }

  cat("^^^^^^^^^^^^^00000000000000000000000 df_metrics ^^^^^^^^^^^^^00000000000000000000000\n")
  # 1.2 get df with perf metrics [per val_strat/MDS]
  df_metrics = NULL
  if(1 == 1) {
    for(mds in uniq(all_preds_df$MDS)) {
      for(val_strat in uniq(all_preds_df$val_strat)){
        print(sprintf("%s + %s", mds, val_strat))
        print("*******")
        c_df = all_preds_df[all_preds_df$MDS == mds,]
        c_df = c_df[c_df$val_strat == val_strat,]
        if(nrow(c_df) == 0) {
          print("EMPTY ... SKIPPING !")
          next
        }
        c_res = sorted_mean_reses[sorted_mean_reses$model_tp == mds,]
        c_res = c_res[c_res$val_strat == val_strat,]
        c_res_auc = c_res[c_res$metric =='AUC-ROC',]
        c_res_cint = c_res[c_res$metric =='calibration-intercept',]
        c_res_cslp = c_res[c_res$metric =='calibration-slope',]

        c_metrics = as.data.frame(list(
          model_nm = c_res_auc$model_tp,
          center = 'mean',
          auc = c_res_auc$mean,
          auc_95lo = c_res_auc$ci95.lower,
          auc_95hi = c_res_auc$ci95.upper,
          cali_int = c_res_cint$mean,
          cali_int_95lo  = c_res_cint$ci95.lower,
          cali_int_95hi = c_res_cint$ci95.upper,
          cali_slp = c_res_cslp$mean,
          cali_slp_95lo = c_res_cslp$ci95.lower,
          cali_slp_95hi = c_res_cslp$ci95.upper,
          val_strat = c_res_auc$val_strat
        ))
        c_metrics$mds_val_tp_group = sprintf("%s-%s", c_res_auc$model_tp, c_res_auc$val_strat)
        df_metrics = rbind(df_metrics, c_metrics)
      }
    }

  }
  #get vars_df *used for getting list of vars selected per model
  vars_df = NULL
  cat("^^^^^^^^^^^^^00000000000000000000000 vars_df ^^^^^^^^^^^^^00000000000000000000000\n")
  if(1==1) {
    for(val_strat in names(RFILES)){
      for(model_type in names(RFILES[[val_strat]])){
        cat(val_strat, model_type, sep = '\t')
        cat('\n')
        c_fn = RFILES[[val_strat]][[model_type]]
        if(c_fn  == -1)
          next
        c_model = load_with_assign(c_fn)
        # Handle each model_type ensemble_glob_recal / fedavg / local / global
        if(model_type == 'ensemble_glob_recal' || model_type == 'ensemble_no_recal') {
          for(i in 1:len(c_model$fitted_models)){
            print(i)
            cc_model = c_model$fitted_models[[i]]
            if(len(cc_model) == 0)
              next
            for(j in 1:len(cc_model$lasso_models_list)) {
              c_local_model = cc_model$lasso_models_list[[j]]
              idxs = which(c_local_model[[1]]$coefficients != 0)
              betas = c_local_model[[1]]$coefficients[idxs]
              c_vars_selected = names(betas)

              c_row = as.data.frame(list(
                val_strat = val_strat,
                model_type = model_type,
                variable = c_vars_selected,
                coefficient  = as.numeric(betas),
                fold = i,
                center_id = class(c_local_model)[2]
              ))
              vars_df = rbind(vars_df, c_row)
            }
            if(is.null(cc_model$lr_from_local_models))
              betas = cc_model$center_weights
            else
              betas = cc_model$lr_from_local_models$coefficients
            betas = betas[which(betas != 0)]
            c_vars_selected = names(betas)
            c_row = as.data.frame(list(
              val_strat = val_strat,
              model_type = model_type,
              variable = c_vars_selected,
              coefficient  = as.numeric(betas),
              fold = i,
              center_id = class(c_local_model)[3]
            ))
            vars_df = rbind(vars_df, c_row)
          }
        }
        if(model_type == 'fedavg_no_recal' || model_type == 'fedavg_glob_recal') {
          for(i in 1:len(c_model$fitted_models)) {
            print(i)
            cc_model = c_model$fitted_models[[i]]
            betas = cc_model$coefficients
            betas = betas[which(betas != 0)]
            c_vars_selected = names(betas)
            if(len(c_vars_selected) != 0) {
              c_row = as.data.frame(list(
                val_strat = val_strat,
                model_type = model_type,
                variable = c_vars_selected,
                coefficient  = as.numeric(betas),
                fold = i,
                center_id = 'NA'
              ))
              vars_df = rbind(vars_df, c_row)
            }
          }
        }
        if(model_type == 'local') {
          for(i in 1:len(c_model$fitted_models)){
            print(i)
            cc_model = c_model$fitted_models[[i]]
            for(j in 1:len(cc_model)) {
              c_local_model = cc_model[[j]]
              idxs = which(c_local_model[[1]]$coefficients != 0)
              betas = c_local_model[[1]]$coefficients[idxs]
              c_vars_selected = names(betas)
              c_row = as.data.frame(list(
                val_strat = val_strat,
                model_type = model_type,
                variable = c_vars_selected,
                coefficient  = as.numeric(betas),
                fold = i,
                center_id = class(c_local_model)[2]
              ))
              vars_df = rbind(vars_df, c_row)
            }
          }
        }
        if(model_type == 'global') {
          for(i in 1:len(c_model$fitted_models)){
            print(i)
            cc_model = c_model$fitted_models[[i]]
            c_local_model = cc_model
            idxs = which(c_local_model$coefficients != 0)
            betas = c_local_model$coefficients[idxs]
            c_vars_selected = names(c_local_model$coefficients)[idxs]
            if(len(c_vars_selected) == 0)
              next
            c_row = as.data.frame(list(
              val_strat = val_strat,
              model_type = model_type,
              variable = c_vars_selected,
              coefficient  = as.numeric(betas),
              fold = i, center_id = i
            ))
            vars_df = rbind(vars_df, c_row)
          }
        }
      }
    }
  }
}

##### PART 1: AUC_TEST_SIG  #####

with_bootstrap = F #boostrap takes a long time, delong always agrees with bootstrap, use delong firs to get results while waiting
save_xlsx = F
if(AUC_TEST_SIG) {
  roc_methods = c("delong") # DeLong test for AUC runs much faster and so far has always agreed with bootstrap results, use to run quick when debugging
  if(with_bootstrap)
    roc_methods = c(roc_methods, "bootstrap")

  all_preds_lcoa = all_preds_df[all_preds_df$val_strat=="LCOA",]
  all_preds_10cv = all_preds_df[all_preds_df$val_strat=="CV",]

  #lcoa
  mdss = c("global", "fedavg_no_recal", "fedavg_glob_recal", "ensemble_no_recal", "ensemble_glob_recal")
  all_methods_lcoa_auc_diffs = NULL
  for(roc_method in roc_methods) {
    # test auc diffs per center
    lcoa_auc_diffs = NULL
    for(c_c in c(1:16)) {
      for(i in 1:(len(mdss)-1) ){
        for(j in (i+1):len(mdss)){
          c_mds1 = mdss[i]
          c_mds2 = mdss[j]
          c_df1 = all_preds_lcoa[all_preds_lcoa$MDS == c_mds1,]
          if(c_c != "ALL")
            c_df1 = c_df1[c_df1$center == c_c,]
          c_df1 = c_df1[c_df1$fold != 17,] # remove final model
          if(nrow(c_df1) == 0)
            next
          c_roc1 = suppressMessages(pROC::roc( c_df1$outcome, c_df1$prediction, direction="<" ))
          c_df2 = all_preds_lcoa[all_preds_lcoa$MDS == c_mds2,]
          if(c_c != "ALL")
            c_df2 = c_df2[c_df2$center == c_c,]
          c_df2 = c_df2[c_df2$fold != 17,] # remove final model
          if(nrow(c_df2) == 0)
            next
          c_roc2 = suppressMessages(pROC::roc( c_df2$outcome, c_df2$prediction, direction="<" ))
          c_diff = pROC::roc.test(c_roc1, c_roc2, paired = pROC::are.paired(c_roc1, c_roc2), method = roc_method)
          is_better = c_diff$p.value <= 0.05 && c_diff$estimate[[1]] > c_diff$estimate[[2]]
          is_worse = c_diff$p.value <= 0.05 && c_diff$estimate[[1]] < c_diff$estimate[[2]]
          is_sig_diff =  is_better || is_worse
          c_row = as.data.frame(list(fold = c_c,
                                     m1 = c_mds1,
                                     m2 = c_mds2,
                                     pval = c_diff$p.value,
                                     est1 = c_diff$estimate[[1]],
                                     est2 = c_diff$estimate[[2]],
                                     winner = if(is_better) c_mds1 else if(is_worse) c_mds2 else "NONE",
                                     loser = if(is_better) c_mds2 else if(is_worse) c_mds1 else "NONE"))
          lcoa_auc_diffs = rbind(lcoa_auc_diffs, c_row)
        }
      }
    }
    lcoa_auc_diffs = lcoa_auc_diffs[order(lcoa_auc_diffs$m1, lcoa_auc_diffs$m2, lcoa_auc_diffs$fold ),]
    all_methods_lcoa_auc_diffs[[roc_method]] = lcoa_auc_diffs
    if(save_xlsx)
      writexl::write_xlsx(lcoa_auc_diffs, sprintf("./tagged-xlsx/%s_lcoa-auc-diff-test-method-%s-per-fold.xlsx", SENS_ANALYSIS_FN_PREFIX, roc_method) )
  }

  #10CV
  mdss = c("global", "fedavg_no_recal", "fedavg_glob_recal", "ensemble_no_recal", "ensemble_glob_recal", 'local')
  all_methods_10cv_auc_diffs = NULL
  for(roc_method in roc_methods) {
    cv10_auc_diffs = NULL
    for(c_c in c(1:10)) {
      for(i in 1:(len(mdss)-1)){
        for(j in (i+1):len(mdss)){
          c_mds1 = mdss[i]
          c_mds2 = mdss[j]
          c_df1 = all_preds_10cv[all_preds_10cv$MDS == c_mds1,]
          if(c_c != "ALL")
            c_df1 = c_df1[c_df1$fold == c_c,]
          c_df1 = c_df1[c_df1$fold != 11,] # remove final model
          if(nrow(c_df1) == 0)
            next
          c_roc1 = suppressMessages(pROC::roc( c_df1$outcome, c_df1$prediction, direction="<" ))
          c_df2 = all_preds_10cv[all_preds_10cv$MDS == c_mds2,]
          if(c_c != "ALL")
            c_df2 = c_df2[c_df1$fold == c_c,]
          c_df2 = c_df2[c_df2$fold != 11,] # remove final model
          if(nrow(c_df2) == 0)
            next
          c_roc2 = suppressMessages(pROC::roc( c_df2$outcome, c_df2$prediction, direction="<" ))
          c_diff = pROC::roc.test(c_roc1, c_roc2, paired = pROC::are.paired(c_roc1, c_roc2),
                                  method = roc_method, boot.n=3000)
          is_better = c_diff$p.value <= 0.05 && c_diff$estimate[[1]] > c_diff$estimate[[2]]
          is_worse = c_diff$p.value <= 0.05 && c_diff$estimate[[1]] < c_diff$estimate[[2]]
          is_sig_diff =  is_better || is_worse
          c_row = as.data.frame(list(fold = c_c,
                                     m1 = c_mds1,
                                     m2 = c_mds2,
                                     pval = c_diff$p.value,
                                     est1 = c_diff$estimate[[1]],
                                     est2 = c_diff$estimate[[2]],
                                     winner = if(is_better) c_mds1 else if(is_worse) c_mds2 else "NONE",
                                     loser = if(is_better) c_mds2 else if(is_worse) c_mds1 else "NONE"))
          cv10_auc_diffs = rbind(cv10_auc_diffs, c_row)

        }

      }
    }
    cv10_auc_diffs = cv10_auc_diffs[order(cv10_auc_diffs$m1, cv10_auc_diffs$m2, cv10_auc_diffs$fold ),]
    all_methods_10cv_auc_diffs[[roc_method]] = cv10_auc_diffs
    if(save_xlsx)
      writexl::write_xlsx(cv10_auc_diffs, sprintf("./tagged-xlsx/%s_10cv-auc-diff-test-method-%s-per-fold.xlsx", SENS_ANALYSIS_FN_PREFIX, roc_method) )
  }

  if(!save_xlsx) {
    View(cv10_auc_diffs[cv10_auc_diffs$winner != 'NONE',])
    View(lcoa_auc_diffs[lcoa_auc_diffs$winner != 'NONE',])
  }

}


##### PART 2: REMA ####
if(RUN_REMA_MEANS) {
  View(sorted_mean_reses)
  writexl::write_xlsx(sorted_mean_reses, sprintf("./tagged-xlsx/%s_mean_meta_reses-%s.xlsx",
                                                 SENS_ANALYSIS_FN_PREFIX, get_current_datetime_filename_formatted()) )

}

##### PART 3: PER CENTER METRICS ####
# View(per_center_metrics)
if(SAVE_XLSX_per_center_metrics)
  writexl::write_xlsx(per_center_metrics, sprintf("./tagged-xlsx/%s_per_center_metrics-%s.xlsx",
                                                  SENS_ANALYSIS_FN_PREFIX, get_current_datetime_filename_formatted()) )
##### PART 4: OVERLAY MODEL PERF PLOTS  #####
if(GEN_PLOTS) {
  ##### init #####
  all_preds_df_cpy = all_preds_df
  # shorter names to match from sorted_mean_reses
  all_preds_df[all_preds_df$MDS == "global",]$model_name = 'global'
  all_preds_df[all_preds_df$MDS == "fedavg_no_recal",]$model_name = 'fedavg_no_recal'
  all_preds_df[all_preds_df$MDS == "fedavg_glob_recal",]$model_name = 'fedavg_glob_recal'
  all_preds_df[all_preds_df$MDS == "ensemble_glob_recal",]$model_name = 'ensemble_glob_recal'
  all_preds_df[all_preds_df$MDS == "ensemble_no_recal",]$model_name = 'ensemble_no_recal'
  all_preds_df[all_preds_df$MDS == "local",]$model_name = 'local'

  metrics_nms = c('auc', 'cali_int', 'cali_slp')
  group_nms = uniq(df_metrics$mds_val_tp_group)
  plot_df = NULL
  for(metrics_nm in metrics_nms) {
    for(group_nm in group_nms) {
      c_val = df_metrics[df_metrics$mds_val_tp_group == group_nm, metrics_nm]
      c_cilo = df_metrics[df_metrics$mds_val_tp_group == group_nm, sprintf("%s_95lo", metrics_nm)]
      c_cihi = df_metrics[df_metrics$mds_val_tp_group == group_nm, sprintf("%s_95hi", metrics_nm)]
      c_row = as.data.frame(list(group_nm = group_nm, metric_nm = metrics_nm,
                                 metric_val = c_val, metric_cilo = c_cilo, metric_cihi = c_cihi))
      plot_df = rbind(plot_df, c_row)
    }
  }

  # Nicer names for metrics
  plot_df[plot_df$metric_nm == 'auc',]$metric_nm = 'AUC-ROC'
  plot_df[plot_df$metric_nm == 'cali_slp',]$metric_nm = 'Calibration Slope'
  plot_df[plot_df$metric_nm == 'cali_int',]$metric_nm = 'Calibration Intercept'
  val_strats = c("Leave Center Out Analysis (LCOA)", "Cross-Validation (CV)")
  plot_df$val_strat = val_strats[1]
  # rename stuff to nicer labels
  if('global-LCOA' %in% plot_df$group_nm)
    plot_df[plot_df$group_nm == 'global-LCOA',]$group_nm = 'Central'
  if('local-LCOA' %in% plot_df$group_nm)
    plot_df[plot_df$group_nm == 'local-LCOA',]$group_nm = 'Local'

  if('ensemble_glob_recal-LCOA' %in% plot_df$group_nm)
    plot_df[plot_df$group_nm == 'ensemble_glob_recal-LCOA',]$group_nm = 'Ensemble'
  if('ensemble_no_recal-LCOA' %in% plot_df$group_nm)
    plot_df[plot_df$group_nm == 'ensemble_no_recal-LCOA',]$group_nm = 'Ensemble'

  if('fedavg_glob_recal-LCOA' %in% plot_df$group_nm)
    plot_df[plot_df$group_nm == 'fedavg_glob_recal-LCOA',]$group_nm = 'FedAvg'
  if('fedavg_no_recal-LCOA' %in% plot_df$group_nm)
    plot_df[plot_df$group_nm == 'fedavg_no_recal-LCOA',]$group_nm = 'FedAvg'


  if('global-CV' %in% plot_df$group_nm){
    plot_df[plot_df$group_nm == 'global-CV',]$val_strat = val_strats[2]
    plot_df[plot_df$group_nm == 'global-CV',]$group_nm = 'Central'
  }
  if('local-CV' %in% plot_df$group_nm) {
    plot_df[plot_df$group_nm == 'local-CV',]$val_strat = val_strats[2]
    plot_df[plot_df$group_nm == 'local-CV',]$group_nm = 'Local'
  }
  if('ensemble_glob_recal-CV' %in% plot_df$group_nm) {
    plot_df[plot_df$group_nm == 'ensemble_glob_recal-CV',]$val_strat = val_strats[2]
    plot_df[plot_df$group_nm == 'ensemble_glob_recal-CV',]$group_nm = 'Ensemble'
  }
  if('ensemble_no_recal-CV' %in% plot_df$group_nm) {
    plot_df[plot_df$group_nm == 'ensemble_no_recal-CV',]$val_strat = val_strats[2]
    plot_df[plot_df$group_nm == 'ensemble_no_recal-CV',]$group_nm = 'Ensemble'
  }
  if('fedavg_no_recal-CV' %in% plot_df$group_nm) {
    plot_df[plot_df$group_nm == 'fedavg_no_recal-CV',]$val_strat = val_strats[2]
    plot_df[plot_df$group_nm == 'fedavg_no_recal-CV',]$group_nm = 'FedAvg'
  }
  if('fedavg_glob_recal-CV' %in% plot_df$group_nm) {
    plot_df[plot_df$group_nm == 'fedavg_glob_recal-CV',]$val_strat = val_strats[2]
    plot_df[plot_df$group_nm == 'fedavg_glob_recal-CV',]$group_nm = 'FedAvg'
  }

  all_preds_cv = all_preds_df[all_preds_df$val_strat=="CV",]
  table(all_preds_cv$fold)
  table(all_preds_cv$center)

  all_preds_lcoa = all_preds_df[all_preds_df$val_strat=="LCOA",]
  table(all_preds_lcoa$fold)
  table(all_preds_lcoa$center)
  mds_to_lty_mapping = list(FedAvg = 3,
                            Ensemble = 3)

  # AUC roc curves  - ######
  # :: for all centers/folds
  # CV plot

  # no recal
  # svg(sprintf("../plots/perf-res/auc-roc-curves/%s_no_recal_cv.svg", SENS_ANALYSIS_FN_PREFIX),
  #     width = 7, height = 6)
  # plot_aucs_overlay(all_preds_cv, df_metrics, title = "AUC-ROC curves from Cross-validation\n(no recalibration)", val_strat = "CV",
  #                   use_pooled_metric = T, legend_xy = c(0.5, 0.24), skip_MDSs = c('ensemble_glob_recal', 'fedavg_glob_recal'))
  # dev.off()

  mds_to_lty_mapping_auc = mds_to_lty_mapping
  mds_to_lty_mapping_auc$FedAvg = 1
  mds_to_lty_mapping_auc$`Ensemble` = 1

  # yes recal
  svg(sprintf("../plots/perf-res/auc-roc-curves/%s_yes_recal_cv.svg", SENS_ANALYSIS_FN_PREFIX),
      width = 7, height = 6)
  plot_aucs_overlay(all_preds_cv, df_metrics, title = "AUC-ROC curves from Cross-validation\n(with recalibration)", val_strat = "CV",
                    use_pooled_metric = T, legend_xy = c(0.42, 0.24), skip_MDSs = c('ensemble_no_recal', 'fedavg_no_recal'),
                    legend_include_sd = F, mds_to_lty_mapping = mds_to_lty_mapping_auc)

  dev.off()

  # LCOA plot

  # no recal
  # svg(sprintf("../plots/perf-res/auc-roc-curves/%s_no_recal_lcoa.svg", SENS_ANALYSIS_FN_PREFIX),
  #     width = 7, height = 6)
  # plot_aucs_overlay(all_preds_lcoa, df_metrics, title = "AUC-ROC curves from Leave-Center-Out Analysis\n(no recalibration)", val_strat = "LCOA",
  #                   use_pooled_metric = T, legend_xy = c(0.5, 0.24), skip_MDSs = c('ensemble_glob_recal', 'fedavg_glob_recal'))
  # dev.off()

  # yes recal
  svg(sprintf("../plots/perf-res/auc-roc-curves/%s_yes_recal_lcoa.svg", SENS_ANALYSIS_FN_PREFIX),
      width = 7, height = 6)
  plot_aucs_overlay(all_preds_lcoa, df_metrics, title = "AUC-ROC curves from Leave-Center-Out Analysis\n(with recalibration)", val_strat = "LCOA",
                    use_pooled_metric = T, legend_xy = c(0.42, 0.18), skip_MDSs = c('ensemble_no_recal', 'fedavg_no_recal'),
                    legend_include_sd = F,  mds_to_lty_mapping = mds_to_lty_mapping_auc)
  dev.off()


  ####cali plots  ######

  # For all centers
  # LCOA

  # no recal
  # svg(sprintf("../plots/perf-res/cali/%s_no_recal_lcoa.svg", SENS_ANALYSIS_FN_PREFIX),
  #     width = 7, height = 7)
  # plot_cali_overlay(all_preds_lcoa, df_metrics, val_strat = "LCOA", df = 8, scale_preds = F, strat_on ='center',
  #                   extra_title_txt = '\n(no recalibration)', skip_MDSs = c('ensemble_glob_recal', 'fedavg_glob_recal'))
  # dev.off()

  # yes recal
  # svg(sprintf("../plots/perf-res/cali/%s_yes_recal_lcoa.svg", SENS_ANALYSIS_FN_PREFIX),
  #     width = 7, height = 7)
  # plot_cali_overlay(all_preds_lcoa, df_metrics, val_strat = "LCOA", df = 8, scale_preds = F, strat_on ='center',
  #                   extra_title_txt = '\n(with recalibration)', skip_MDSs = c('ensemble_no_recal', 'fedavg_no_recal'))
  # dev.off()

  # cv

  # no recal
  # svg(sprintf("../plots/perf-res/cali/%s_no_recal_cv.svg", SENS_ANALYSIS_FN_PREFIX),
  #     width = 7, height = 7)
  # plot_cali_overlay(all_preds_cv, df_metrics,  val_strat = "CV", df = 8, scale_preds = F, strat_on = 'fold',
  #                   extra_title_txt = '\n(no recalibration)',skip_MDSs = c('ensemble_glob_recal', 'fedavg_glob_recal'))
  # dev.off()
  #
  # # yes recal
  # svg(sprintf("../plots/perf-res/cali/%s_yes_recal_cv.svg", SENS_ANALYSIS_FN_PREFIX),
  #     width = 7, height = 7)
  # plot_cali_overlay(all_preds_cv, df_metrics,  val_strat = "CV", df = 8, scale_preds = F, strat_on = 'fold',
  #                   extra_title_txt = '\n(with recalibration)',skip_MDSs = c('ensemble_no_recal', 'fedavg_no_recal'))
  # dev.off()

  # now with only bottom 97.5% of preds
  # LCOA


  # no recal
  # svg(sprintf("../plots/perf-res/cali/%s_lcoa-top2.5excl_no_recal.svg", SENS_ANALYSIS_FN_PREFIX),
  #     width = 7, height = 7)
  # plot_cali_overlay(all_preds_lcoa, df_metrics, val_strat = "LCOA", excluded_percentiles = c(0,97.5),
  #                   xlim = c(0, 0.18), ylim  = NULL, legend_correction = c(0.055, 0.029), df = 4, strat_on ='center',
  #                   extra_title_txt = '\n(no recalibration)',skip_MDSs = c('ensemble_glob_recal', 'fedavg_glob_recal'),
  #                   mds_to_lty_mapping = mds_to_lty_mapping)
  # dev.off()

  # yes recal
  svg(sprintf("../plots/perf-res/cali/%s_lcoa-top2.5excl_yes_recal.svg", SENS_ANALYSIS_FN_PREFIX),
      width = 7, height = 7)
  plot_cali_overlay(all_preds_lcoa, df_metrics, val_strat = "LCOA", excluded_percentiles = c(0,97.5),
                    xlim = c(0, 0.18), ylim  = NULL, legend_correction = c(0.078, 0.029), df = 4, strat_on ='center',
                    extra_title_txt = '\n(with recalibration)',skip_MDSs = c('ensemble_no_recal', 'fedavg_no_recal'),
                    legend_contents = c('given_int_mean', 'given_slp_mean'),
                    mds_to_lty_mapping = mds_to_lty_mapping)
  dev.off()

  # CV

  # no recal
  # svg(sprintf("../plots/perf-res/cali/%s_cv-top2.5excl_no_recal.svg", SENS_ANALYSIS_FN_PREFIX),
  #     width = 7, height = 7)
  # plot_cali_overlay(all_preds_cv, df_metrics, val_strat = "CV", excluded_percentiles = c(0,97.5),
  #                   xlim = c(0, 0.18), ylim  = NULL, legend_correction = c(0.055, 0.029), df = 4, strat_on ='fold',
  #                   extra_title_txt = '\n(no recalibration)',skip_MDSs = c('ensemble_glob_recal', 'fedavg_glob_recal'))
  # dev.off()

  # yes recal
  svg(sprintf("../plots/perf-res/cali/%s_cv-top2.5excl_yes_recal.svg", SENS_ANALYSIS_FN_PREFIX),
      width = 7, height = 7)
  plot_cali_overlay(all_preds_cv, df_metrics, val_strat = "CV", excluded_percentiles = c(0,97.5),
                    xlim = c(0, 0.18), ylim  = NULL, legend_correction = c(0.079, 0.029), df = 4, strat_on ='fold',
                    extra_title_txt = '\n(with recalibration)',skip_MDSs = c('ensemble_no_recal', 'fedavg_no_recal'),
                    mds_to_lty_mapping = mds_to_lty_mapping,
                    legend_contents = c('given_int_mean', 'given_slp_mean'))
  dev.off()

  #####
  all_preds_df = all_preds_df_cpy
}




##### PART 5: [VARS SELECTED + hyperparams] from final models ####
if(GEN_VARS_SELECTED) {
  save_xlsx = F
  uniq_vars_grouped_df = NULL
  if(1 == 1) {
    for(val_strat in uniq(vars_df$val_strat)){
      print(val_strat)
      b_df = vars_df[vars_df$val_strat == val_strat,]
      for(model_tp in uniq(b_df$model_type)){
        print(model_tp)
        c_df = b_df[b_df$model_type == model_tp,]
        c_uniq_grouped_vars = uniq_and_group_predictors(c_df$variable)
        if(len(grep('center_',c_uniq_grouped_vars)) != 0)
          c_uniq_grouped_vars = c_uniq_grouped_vars[-grep('center_',c_uniq_grouped_vars)]
        c_row = as.data.frame(list(vars = c_uniq_grouped_vars))
        c_row$val_strat = c_df$val_strat[1]
        c_row$model_type = c_df$model_type[1]
        uniq_vars_grouped_df = rbind(uniq_vars_grouped_df, c_row)
      }
    }
    print("uniq_vars_grouped_df = ")
    print(uniq_vars_grouped_df[1:5,])
    all_uniq_vars = sort(uniq(uniq_vars_grouped_df$vars))
    cat(sprintf("%s\n", all_uniq_vars))
    for(i in all_uniq_vars){
      c_rows = which(uniq_vars_grouped_df$vars == i)
      if(i %in% names(vars_grouped_fancy_names_mapping)) {
        uniq_vars_grouped_df[c_rows,]$vars = vars_grouped_fancy_names_mapping[[i]]
      }
      # else print(i)
    }
    if(save_xlsx) {
      print("saving ./tagged-xlsx/uniq_vars_grouped_df.xlsx ....")
      writexl::write_xlsx(uniq_vars_grouped_df, sprintf("./tagged-xlsx/%s_uniq_vars_grouped_df.xlsx", SENS_ANALYSIS_FN_PREFIX) )
      print("saving ./tagged-xlsx/vars_df.xlsx ....")
      writexl::write_xlsx(vars_df, sprintf("./tagged-xlsx/%s_vars_df.xlsx", SENS_ANALYSIS_FN_PREFIX) )
    }
  }

  #hyperparams
  # fedavg
  f_hyper_params = as.data.frame(list(model = c(NA), val_strat = c(NA), param = c(NA), param_val = c(NA)))
  n_row = f_hyper_params
  for(c_vm in c("CV", "LCOA")) {
    n_row$val_strat = c_vm
    # fedavg
    c_m = load_with_assign(RFILES[[c_vm]]$fedavg_glob_recal)
    # todo: note: you are missing the selected lambda hyperparam here!
    # it would have appeared during the variable selection step
    f_m = c_m$fitted_models[[len(c_m$fitted_models)]] # take final model results
    n_row$model = 'fedavg'
    n_row$param = 'lr'
    n_row$param_val = f_m$training_params$learning_rate
    f_hyper_params = rbind(f_hyper_params, n_row)

    n_row$param = 'epochs'
    n_row$param_val = f_m$training_params$epochs
    f_hyper_params = rbind(f_hyper_params, n_row)

    n_row$param = 'agr_str'
    n_row$param_val = f_m$training_params$agrreement_strength_lasso
    f_hyper_params = rbind(f_hyper_params, n_row)

    for(c_coef_idx in 1:len(f_m$coefficients)){
      c_coef_nm = names(f_m$coefficients)[c_coef_idx]
      c_coef = f_m$coefficients[c_coef_idx]
      n_row$param = c_coef_nm
      n_row$param_val = c_coef
      f_hyper_params = rbind(f_hyper_params, n_row)

    }


    # global
    # todo: note: you are missing the selected lambda hyperparam here!
    c_m = load_with_assign(RFILES[[c_vm]]$global)
    f_m = c_m$fitted_models[[len(c_m$fitted_models)]]
    n_row$model = 'global'

    for(c_coef_idx in 1:len(f_m$coefficients)){
      c_coef_nm = names(f_m$coefficients)[c_coef_idx]
      c_coef = f_m$coefficients[c_coef_idx]
      n_row$param = c_coef_nm
      n_row$param_val = c_coef
      f_hyper_params = rbind(f_hyper_params, n_row)
    }

    #stacked
    # todo: note: you are missing the selected lambda hyperparam here!
    c_m = load_with_assign(RFILES[[c_vm]]$ensemble_glob_recal)
    fs_m = c_m$fitted_models[[len(c_m$fitted_models)]]
    n_row$model = 'ensemble_glob_recal'
    # all local models in stacked
    for(f_m_idx in 1:len(fs_m$lasso_models_list)){
      f_m = fs_m$lasso_models_list[[f_m_idx]][[1]]
      for(c_coef_idx in 1:len(f_m$coefficients)){
        c_coef_nm = names(f_m$coefficients)[c_coef_idx]
        c_coef = f_m$coefficients[c_coef_idx]
        n_row$param = sprintf('cnt_%s_%s',f_m_idx, c_coef_nm)
        n_row$param_val = c_coef
        if(is.na(c_coef) || is.null(c_coef))
          next
        f_hyper_params = rbind(f_hyper_params, n_row)
      }
    }
    # top model in stacked
    f_m = fs_m$lr_from_local_models
    # f_hyper_params = list( model = 'ensemble_glob_recal', val_strat = c_vm, cnt = -1)
    for(c_coef_idx in 1:len(f_m$coefficients)){
      c_coef_nm = names(f_m$coefficients)[c_coef_idx]
      c_coef = f_m$coefficients[c_coef_idx]
      if(is.na(c_coef) || is.null(c_coef))
        next
      n_row$param = c_coef_nm
      n_row$param_val = c_coef
      f_hyper_params = rbind(f_hyper_params, n_row)
    }
    #locals
    if(c_vm == 'CV') {
      c_m = load_with_assign(RFILES[[c_vm]]$local)
      fs_m = c_m$fitted_models[[len(c_m$fitted_models)]]
      n_row$model = 'local'
      # all local models in stacked
      for(f_m_idx in 1:len(fs_m)){
        f_m = fs_m[[f_m_idx]][[1]]
        for(c_coef_idx in 1:len(f_m$coefficients)){
          c_coef_nm = names(f_m$coefficients)[c_coef_idx]
          c_coef = f_m$coefficients[c_coef_idx]
          if(is.na(c_coef) || is.null(c_coef))
            next
          n_row$param = sprintf('cnt_%s_%s',f_m_idx, c_coef_nm)
          n_row$param_val = c_coef
          f_hyper_params = rbind(f_hyper_params, n_row)
        }
      }
    }


  }
  View(f_hyper_params[f_hyper_params$model=='ensemble_glob_recal',])
  View(f_hyper_params[f_hyper_params$model=='local',])
  writexl::write_xlsx(f_hyper_params[-1,], sprintf("./tagged-xlsx/%s_f_hyper_and_normal_params.xlsx", SENS_ANALYSIS_FN_PREFIX) )
  # df_cv = as.data.frame(f_hyper_params$CV)
  # df_cv$val_strat = 'CV'
  # df_lcoa = as.data.frame(f_hyper_params$LCOA)
  # df_lcoa$val_strat = 'LCOA'
  # df_params = rbind(df_cv ,df_lcoa)

  # others...
  # c_f ="lambda.min_lambd_fld=7;mdl=lr-lasso-global;vs=CV.RData"
  dir_to_scan = "../data/out/lambdas"
  lambdas_df = NULL
  for(c_f in list.files(dir_to_scan)){
    # cat(sprintf("Reading %s ..\n", c_f))
    c_lmbd = load_with_assign(sprintf("%s/%s",dir_to_scan,c_f))

    match <- regexpr("fld=(/d+;)", c_f)
    c_fold = as.numeric(stringr::str_sub( regmatches(c_f, match) , 5, -2))

    match <- regexpr("mdl=(.+;)", c_f)
    c_model = stringr::str_sub( regmatches(c_f, match) , 5,-2)

    match <- regexpr("vs=(/w+)", c_f)
    c_val_strat = stringr::str_sub( regmatches(c_f, match) , 4,-1)

    match <- regexpr("lambda-(/d+/.)", c_f)
    c_cent = NA
    if(match != -1) {
      c_cent = as.numeric(stringr::str_sub( regmatches(c_f, match) , 8,-2))
    }
    else {
      match <- regexpr("RData_(/d+)$", c_f)
      if(match != -1)
        c_cent = as.numeric(stringr::str_sub( regmatches(c_f, match) , 7,-1))
    }

    n_row = as.data.frame(list(c_val_strat= c_val_strat,  c_model = c_model, c_fold= c_fold, c_cent = c_cent,
                               c_lmbd= c_lmbd  ))
    if(n_row$c_val_strat == 'CV' && n_row$c_model == 'fed-lr-lasso-gradient-update' && n_row$c_cent == 1)
      print(c_f)
    lambdas_df = rbind(lambdas_df, n_row)



  }
  View(lambdas_df)
  table(lambdas_df[,c("c_val_strat", "c_fold", "c_model", "c_cent")])
  writexl::write_xlsx(lambdas_df, sprintf("./tagged-xlsx/%s_f_hyper_lambda-8-aug.xlsx", SENS_ANALYSIS_FN_PREFIX) )


  final_model_nvars_local = sort( c(10, 3, 10, 9, 10, 5, 7, 15, 12, 4, 9, 6, 14, 3) ) - 1
  quantile(final_model_nvars_local, probs = c(0.25, 0.75))

}




##### PART 6: NRI matrix #####
if(SAVE_NRI_MATRIX){

  writexl::write_xlsx(nri_matrix_list$CV$event, sprintf("./tagged-xlsx/%s_nri_matrix_list-CV-event.xlsx",SENS_ANALYSIS_FN_PREFIX) )
  writexl::write_xlsx(nri_matrix_list$CV$noevent, sprintf("./tagged-xlsx/%s_nri_matrix_list-CV-noevent.xlsx",SENS_ANALYSIS_FN_PREFIX) )

  writexl::write_xlsx(nri_matrix_list$LCOA$event, sprintf("./tagged-xlsx/%s_nri_matrix_list-LCOA-event.xlsx",SENS_ANALYSIS_FN_PREFIX) )
  writexl::write_xlsx(nri_matrix_list$LCOA$noevent, sprintf("./tagged-xlsx/%s_nri_matrix_list-LCOA-noevent.xlsx",SENS_ANALYSIS_FN_PREFIX) )

}
##### PART 7: dot-plot predictions of two models #####
if(RUN_DOT_PLOT_MODEL_PREDS) {
  # glob recal
  plot(cv_preds_pr$fedavg_glob_recal, cv_preds_pr$ensemble_glob_recal)
  cv_preds_pr$center = as.numeric(levels(cv_preds_pr$center)[cv_preds_pr$center])

  ggplot(cv_preds_pr, aes(x = fedavg_glob_recal, y = ensemble_glob_recal, color = factor(center))) +
    geom_point() +
    xlim(0, 1) + ylim(0, 1) +
    labs(x = "fedavg_glob_recal", y = "ensemble_glob_recal", color = "center")  + ggtitle("Dotplot of fedavg_glob_recal and ensemble_glob_recal predictions")


  # no recal
  ggplot(cv_preds_pr, aes(x = fedavg_no_recal, y = ensemble_no_recal, color = factor(center))) +
    geom_point() +
    xlim(0, 1) + ylim(0, 1) +
    labs(x = "fedavg_no_recal", y = "ensemble_no_recal", color = "center")  + ggtitle("Dotplot of fedavg_no_recal and ensemble_no_recal predictions")


  # CV
  # stacked vs stacked
  ggplot(cv_preds_pr, aes(x = ensemble_glob_recal, y = ensemble_no_recal, color = factor(center))) +
    geom_point() +
    xlim(0, 1) + ylim(0, 1) +
    labs(x = "ensemble_glob_recal", y = "ensemble_no_recal", color = "center")  + ggtitle("Dotplot of ensemble_glob_recal and ensemble_no_recal predictions")

  # fedavg vs fedavg
  ggplot(cv_preds_pr, aes(x = fedavg_glob_recal, y = fedavg_no_recal, color = factor(center))) +
    geom_point() +
    xlim(0, 1) + ylim(0, 1) +
    labs(x = "fedavg_glob_recal", y = "fedavg_no_recal", color = "center")  + ggtitle("Dotplot of fedavg_glob_recal and fedavg_no_recal predictions")



  # global vs local (0s only)
  cv_preds_pr_0s = cv_preds_pr[cv_preds_pr$outcome == 0,]
  cv_preds_pr_0s = cv_preds_pr_0s[!is.na(cv_preds_pr_0s$local),]
  cv_preds_pr_0s = cv_preds_pr_0s[sample(1:nrow(cv_preds_pr_0s), 300),]
  # cv_preds_pr_0s$global = try_round(cv_preds_pr_0s$global, 2)
  # cv_preds_pr_0s$local = try_round(cv_preds_pr_0s$local, 2)
  ggplot(cv_preds_pr_0s, aes(x = global, y = local, color = factor(center))) +
    geom_point() +
    xlim(0, 0.25) + ylim(0, 0.25) +
    labs(x = "global", y = "local", color = "center")  + ggtitle("Dotplot of global and local predictions")

  # global vs local (1s only)
  cv_preds_pr_1s = cv_preds_pr[cv_preds_pr$outcome == 1,]
  cv_preds_pr_1s = cv_preds_pr_1s[!is.na(cv_preds_pr_1s$local),]
  cv_preds_pr_1s = cv_preds_pr_1s[sample(1:nrow(cv_preds_pr_1s), 300),]
  ggplot(cv_preds_pr_1s, aes(x = global, y = local, color = factor(center))) +
    geom_point() +
    xlim(0, 0.55) + ylim(0, 0.55) +
    labs(x = "global", y = "local", color = "center")  + ggtitle("Dotplot of global and local predictions")




}



##### PART 8: aesthetics and printing fullnames for variables #####
if(1 == 0) {

  vars_grouped_fancy_names_mapping  = list(
    "aklepchir_eerder1" =  "Previous aortic valve surgery",
    "art_vaatpathologie1" =  "Extra-cardiac arteriopathy",
    "bmi" =  "Body Mass Index (BMI)",
    "bsa" =  "Body Surface Area (BSA)",
    "cardiochir_eerder1" =  "Previous cardiac surgery",
    "CCS_IV1" =  "Canadian Cardiovascular Society (CCS) class IV angina",
    "chr_longziekte1" =  "Chronic lung disease",
    "CVA_eerder1" =  "Previous Cerebrovascular Accident (CVA)",
    "diabetesDM_y" =  "Diabetes Mellitus",
    "dialyse1" =  "Dialysis",
    "eGFR" =  "Estimated glomerular filtration rate",
    "geslacht1" =  "Sex",
    "kreatinine_gehalte" =  "Serum creatinine",
    "leeftijd" =  "Age",
    "narcose1" =  "Anesthesia",
    "krit_preop_toestand1" =  "Critical preoperative state",
    "LVEF" =  "Left Ventricular Ejection Fraction",
    "neuro_disfunctie1" =  "Neurological dysfunction",
    "NYHA2" =  "Functional New York Heart Association (NYHA) class II",
    "NYHA3" =  "Functional New York Heart Association (NYHA) class III",
    "NYHA4" =  "Functional New York Heart Association (NYHA) class IV",
    "PM_eerder1" =  "Previous permanent pacemaker",
    "PA_druk" =  "Pulmonary artery pressure",
    "TAVI_predilatatie1" =  "Balloon pre-TAVI",
    "TAVI_postdilatatie1" =  "Percutaneous Aortic Balloon Valvuloplasty (PABV)",
    "slechte_mobiliteit1" =  "Poor mobility",
    "recent_MI1" =  "Recent Myocardial Infarction (MI)",
    "TAVI_toegangtransfemoral" =  "Access route transfemoral",
    "TAVI_toegangdirect_aortic" =  "Access route direct aortic",
    "TAVI_toegangsubclavian" = "Access route subclavian",
    "TAVI_toegangtransapical" = "Access route transapical",
    "urgentieurgent" =  "Procedure acuity urgent",
    "(Intercept)" = "Intercept")

  cont_cols = c('bmi', 'bsa', 'eGFR', 'kreatinine_gehalte', 'leeftijd', 'LVEF', 'PA_druk', '(Intercept)')

  for(cent_num in 1:16) {
    c_cnt_column = sprintf("center_%d", cent_num)
    cent_let = hidden_center_mapping[[cent_num]]
    vars_grouped_fancy_names_mapping[[c_cnt_column]] = sprintf("Center %s", cent_let)
  }

  final_vars_short_names = c('TAVI_toegangdirect_aortic', 'TAVI_toegangsubclavian', 'TAVI_toegangtransapical', 'TAVI_toegangtransfemoral',
                               'leeftijd', 'narcose1', 'TAVI_predilatatie1', 'bsa', 'CCS_IV1', 'chr_longziekte1', 'krit_preop_toestand1', 'dialyse1', 'art_vaatpathologie1', 'NYHA2', 'NYHA3', 'NYHA4', '(Intercept)', 'LVEF', 'aklepchir_eerder1', 'CVA_eerder1', 'PM_eerder1', 'urgentieurgent', 'PA_druk', 'recent_MI1', 'kreatinine_gehalte', 'TAVI_toegangdirect_aortic', 'TAVI_toegangsubclavian', 'TAVI_toegangtransapical', 'TAVI_toegangtransfemoral', 'leeftijd', 'narcose1', 'TAVI_predilatatie1', 'bmi', 'bsa', 'chr_longziekte1', 'diabetesDM_y', 'eGFR', 'art_vaatpathologie1', 'NYHA2', 'NYHA3', 'NYHA4', '(Intercept)', 'LVEF', 'TAVI_postdilatatie1', 'slechte_mobiliteit1', 'cardiochir_eerder1', 'CVA_eerder1', 'urgentieurgent', 'PA_druk', 'kreatinine_gehalte', 'geslacht1', 'TAVI_toegangdirect_aortic', 'TAVI_toegangtransfemoral', 'leeftijd', 'narcose1', 'bmi', 'chr_longziekte1', 'eGFR', '(Intercept)', 'LVEF', 'urgentieurgent', 'leeftijd', 'narcose1', 'TAVI_predilatatie1', 'bmi', 'bsa', 'chr_longziekte1', 'diabetesDM_y', 'art_vaatpathologie1', '(Intercept)', 'LVEF', 'TAVI_postdilatatie1', 'slechte_mobiliteit1', 'PA_druk', 'kreatinine_gehalte', 'geslacht1', 'leeftijd', 'TAVI_predilatatie1', 'bmi', 'chr_longziekte1', 'diabetesDM_y', 'art_vaatpathologie1', '(Intercept)', 'cardiochir_eerder1', 'urgentieurgent', 'PA_druk', 'TAVI_toegangtransapical', 'TAVI_toegangtransfemoral', 'leeftijd', 'narcose1', 'TAVI_predilatatie1', 'bsa', 'eGFR', 'art_vaatpathologie1', '(Intercept)', 'LVEF', 'TAVI_postdilatatie1', 'cardiochir_eerder1', 'CVA_eerder1', 'PA_druk', 'leeftijd', 'bsa', 'eGFR', 'art_vaatpathologie1', '(Intercept)', 'LVEF', 'PA_druk', 'leeftijd', 'bmi', 'bsa', 'art_vaatpathologie1', 'NYHA3', '(Intercept)', 'urgentieurgent', 'PA_druk', 'kreatinine_gehalte', 'TAVI_toegangtransapical', 'TAVI_toegangtransfemoral', 'leeftijd', 'narcose1', 'TAVI_predilatatie1', 'bmi', 'bsa', 'NYHA2', 'NYHA3', '(Intercept)', 'LVEF', 'geslacht1', 'TAVI_toegangsubclavian', 'TAVI_toegangtransfemoral', 'leeftijd', 'bmi', 'eGFR', 'NYHA3', '(Intercept)', 'slechte_mobiliteit1', 'urgentieurgent', 'kreatinine_gehalte', 'TAVI_toegangtransfemoral', 'bsa', 'NYHA3', '(Intercept)', 'LVEF', 'kreatinine_gehalte', 'TAVI_toegangtransfemoral', 'TAVI_predilatatie1', 'bmi', 'chr_longziekte1', 'NYHA3', 'NYHA4', '(Intercept)', 'PA_druk', 'geslacht1', '(Intercept)', 'LVEF', 'kreatinine_gehalte', 'bmi', 'eGFR', '(Intercept)', 'TAVI_toegangdirect_aortic', 'TAVI_toegangtransfemoral', 'bsa', '(Intercept)', 'leeftijd', 'bsa', '(Intercept)', 'LVEF', 'kreatinine_gehalte', 'TAVI_toegangdirect_aortic', 'TAVI_toegangtransfemoral', 'leeftijd', 'narcose1', 'bmi', 'chr_longziekte1', 'eGFR', '(Intercept)', 'LVEF', 'urgentieurgent', 'narcose1', 'TAVI_predilatatie1', 'bmi', 'bsa', 'chr_longziekte1', 'diabetesDM_y', 'art_vaatpathologie1', '(Intercept)', 'LVEF', 'TAVI_postdilatatie1', 'slechte_mobiliteit1', 'PA_druk', 'kreatinine_gehalte', 'geslacht1', 'leeftijd', 'TAVI_predilatatie1', 'bmi', 'chr_longziekte1', 'diabetesDM_y', 'art_vaatpathologie1', '(Intercept)', 'cardiochir_eerder1', 'urgentieurgent', 'PA_druk', 'TAVI_toegangtransapical', 'TAVI_toegangtransfemoral', 'leeftijd', 'narcose1', 'TAVI_predilatatie1', 'bsa', 'eGFR', 'art_vaatpathologie1', '(Intercept)', 'LVEF', 'TAVI_postdilatatie1', 'cardiochir_eerder1', 'CVA_eerder1', 'PA_druk', 'leeftijd', 'bsa', 'eGFR', 'art_vaatpathologie1', '(Intercept)', 'LVEF', 'PA_druk', 'leeftijd', 'bmi', 'bsa', 'art_vaatpathologie1', 'NYHA3', '(Intercept)', 'urgentieurgent', 'PA_druk', 'kreatinine_gehalte', 'TAVI_toegangtransapical', 'TAVI_toegangtransfemoral', 'leeftijd', 'narcose1', 'TAVI_predilatatie1', 'bsa', 'NYHA2', 'NYHA3', '(Intercept)', 'LVEF', 'geslacht1', 'TAVI_toegangsubclavian', 'TAVI_toegangtransfemoral', 'leeftijd', 'bmi', 'eGFR', 'NYHA3', '(Intercept)', 'slechte_mobiliteit1', 'urgentieurgent', 'kreatinine_gehalte', 'TAVI_toegangtransfemoral', 'eGFR', '(Intercept)', 'kreatinine_gehalte', 'TAVI_toegangtransfemoral', 'TAVI_predilatatie1', 'bmi', 'chr_longziekte1', 'NYHA3', 'NYHA4', '(Intercept)', 'PA_druk', 'geslacht1', 'TAVI_toegangtransfemoral', '(Intercept)', 'LVEF', 'kreatinine_gehalte', 'bmi', 'eGFR', '(Intercept)', 'TAVI_toegangdirect_aortic', 'TAVI_toegangtransfemoral', 'leeftijd', 'bsa', '(Intercept)', 'leeftijd', 'bsa', '(Intercept)', 'LVEF', 'kreatinine_gehalte', 'center_3', 'center_10', 'center_1', 'center_15', 'center_9', 'center_5', 'center_11', 'center_7', '
                               center_14', 'center_13', 'center_16', 'center_2', 'center_12', 'center_8', '(Intercept)' )
  final_vars_long_names = transform_strings_via_mapping(final_vars_short_names, vars_grouped_fancy_names_mapping)
  intersect( uniq(final_vars_long_names), uniq(final_vars_short_names))
  uniq(final_vars_long_names)

  center_ids =  c('center_1', 'center_10', 'center_11', 'center_12', 'center_13', 'center_14', 'center_14', 'center_2', 'center_3', 'center_4', 'center_5', 'center_6', 'center_7', 'center_8', 'center_9', 'center_1', 'center_10', 'center_11', 'center_12', 'center_13', 'center_14', 'center_15', 'center_16', 'center_2', 'center_3', 'center_4', 'center_5', 'center_7', 'center_8', 'center_9', 'NA', 'center_1', 'center_10', 'center_11', 'center_12', 'center_13', 'center_14', 'center_14', 'center_2', 'center_3', 'center_4', 'center_5', 'center_6', 'center_7', 'center_8', 'center_9')
  center_ids = transform_strings_via_mapping(center_ids, vars_grouped_fancy_names_mapping)

  cont_cols_yn = umap(final_vars_short_names, function(x) {if(x %in% cont_cols) "YES" else "NO"})
  cat(paste0(cont_cols_yn,'\n'))
}

