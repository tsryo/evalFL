# Author: T.Y.
# Date created: 26.09.2022


#### working dir set ####
setwd(sprintf("%s/..",dirname(rstudioapi::getSourceEditorContext()$path)))
getwd()
working_dir_ok = readline(sprintf("Confirm working directory (%s) is set to project base dir?[Y/n]: ", getwd()))
if(working_dir_ok != "" && working_dir_ok != "Y" && working_dir_ok != "y")
  setwd(readline("Please enter working dir path (no need to escape \'\\\'): "))

#### imports ####
source("../commons/src/orchestrator.R")
source("src/fl_orchestrator.R")

#### config params ####
RUN_CODE = T # nothing will be ran if this is False

INPUT_DF_PATH = NULL # if no input dataframe file is given, synthetic data will be used instead  #  "../data/df_new.RData"

VALIDATION_STRAT = c("LCOA", "CV")[2]

RUN_HYPERPARAM_OPTIMIZATION <<- T

WITH_SINK = F   # write output to log file?

RUN_GLOBAL = F  # run centralized model strategy?
RUN_LOCAL = T   # run local-only model strategy?
RUN_ENSEMBLE = F  # run (mean) ensemble model strategy?
RUN_FEDERATED = F # run FedAvg model strategy?



N_FOLDS_CV = 10
CAP_KREATININE_EGFR = F #
SAVE_MODELS = T   # save each model (fitted + predictions + params) to a file

SKIP_FOLDS = c(-1) # -1 = dont skip any

HYPERPARAM_METRIC <<- 'inv.auc' # inv.auc, mse
LOG_TRANSFORM_3VARS = F # screat / egfr / sPAP

EXCLUDE_CENTERS_WITH_LESS_THAN_N_TAVIS_AFTER_FIRST_YEAR = 0

# RECALIBRATION
FEDAVG_PERFORM_L1_RECALL = T
FEDAVG_L1_RECALL_STRATEGY = 'GLOBAL' # 'FEDERATED'

STACKED_PERFORM_L1_RECALL = T # GLOBAL ONLY IMPLEMENTED
STACKED_PERFORM_L2_RECALL = F # GLOBAL ONLY IMPLEMENTED # *Deprecated* we do not care about L2 recal really

# hyper-param caching [ only used when RUN_HYPERPARAM_OPTIMIZATION == F ]
FEDAVG_HYPERPARAMS_DICT = list(
  # sensitivity analysis 1 run
  EXCLUDE_CENTERS_WLTNTs_True = list(
    CV = list( LR = rep(0.05, N_FOLDS_CV),
               AGSL = rep(0.5, N_FOLDS_CV) ),
    LCOA =list(LR = c(),
               AGSL = c() ) ),
  # the vanilla run
  EXCLUDE_CENTERS_WLTNTs_False = list(
    CV = list( LR = rep(0.05, N_FOLDS_CV),
               AGSL = rep(0.5, N_FOLDS_CV)),
    LCOA = list( LR = c(),
                 AGSL = c() ) ) )

# pre-computed hyperparams based on sensitivity analysis or main
key_accessor = if(EXCLUDE_CENTERS_WITH_LESS_THAN_N_TAVIS_AFTER_FIRST_YEAR) "EXCLUDE_CENTERS_WLTNTs_True" else "EXCLUDE_CENTERS_WLTNTs_False"
FEDAVG_LR_10CV <<- FEDAVG_HYPERPARAMS_DICT[[key_accessor]]$CV$LR
FEDAVG_AGSLS_10CV <<- FEDAVG_HYPERPARAMS_DICT[[key_accessor]]$CV$AGSL
FEDAVG_LR_LCOA <<- FEDAVG_HYPERPARAMS_DICT[[key_accessor]]$LCOA$LR
FEDAVG_AGSLS_LCOA <<- FEDAVG_HYPERPARAMS_DICT[[key_accessor]]$LCOA$AGSL

HYPERPARAM_LAMBDA_FN <<- "" # filename for storing lamba LASSO hyperparam values


##### debug params #####
ONLY_VAR_SELECT <<- F # stop after selecting variables
subsample_data = -1 # -1 = no subsample // 0.3 subsample gives auc about 0.66 for FL, no subsample gives auc 0.69
DEBUG_ON_EVERY_N_EPOCHS = 9999999 # for FedAvg
plot_hist_every_n_epochs = 9999999 # for FedAvg

###### TRAIN_PARAMS_FL ######

TRAINING_PARAMS_FL = list(           learning_rate =  c(0.01, 0.05, 0.1),
                                     epochs = c(50, 200, 500, 1000),
                                     aggr_technique = "fedavg",  # "average", "fedavg"
                                     agrreement_strength_lasso = c(0, 0.25, 0.5, 0.75),
                                     plot_hist_every_n_epochs = plot_hist_every_n_epochs,
                                     use_ridge = F,
                                     early_stop_delta = 0.00001,
                                     n_folds_cv_glmnet = 5, # with fewer folds, seems to select more variables
                                     cv_glmnet_err_type = 'auc', # = c("default", "mse", "deviance", "class", "auc", "mae", "C"),
                                     lambda_selection_fn = NULL, # deprecated
                                     class_weighting =  F,   # deprecated / no weighting now
                                     err_fn = inv_auc, #mse, #mse_inv_auc_combo,
                                     vars_selected = c(), # deprecated
                                     test_df = NULL,
                                     verbose = T)

## set filenames #############
FILENAME_PREFIX = sprintf("./res_%s_G%i-L%i-FA%i-SE%i-%s", VALIDATION_STRAT, RUN_GLOBAL, RUN_LOCAL,
                          RUN_FEDERATED, RUN_ENSEMBLE, format(Sys.time(), "%m-%d-%H-%M-%Y"))
SINK_NAME = sprintf("%s.log", FILENAME_PREFIX)
results_xlsx_filename = sprintf("%s_res.xlsx", FILENAME_PREFIX)
preds_data_filename = sprintf("%s_preds.RData", FILENAME_PREFIX)

## check config params ############
if(STACKED_PERFORM_L2_RECALL && STACKED_PERFORM_L1_RECALL) {
  try_log_error("Cannot have both STACKED_PERFORM_L2_RECALL and  STACKED_PERFORM_L1_RECALL as True!")
  exit(-1)
}

######
# - workflow
# 1) load & clean data
########## init #######
df1 = NULL
if(is.null(INPUT_DF_PATH)) {
  try_log_warn("No INPUT_DF_PATH provided! Going to use synthetic data instead...")
  df1 = try_gen_synthetic_TAVI_data()
} else {
  try_log_info("Loading dataset from %s...", INPUT_DF_PATH)
  df1 = load_with_assign(INPUT_DF_PATH)
}
df1[,"outcome_of_interest"] = df1[, outcome.var] # rename outcome var to outcome_of_interest
df1[, outcome.var] = NULL
if(EXCLUDE_CENTERS_WITH_LESS_THAN_N_TAVIS_AFTER_FIRST_YEAR > 0){
  remove_centers = umap(uniq(df1[, UOF_VN]), function(x) { c_df = df1[df1[, UOF_VN] == x,]
    if(any(table(c_df$jaar)[(which(table(c_df$jaar) > 0)[1]+1):len(table(c_df$jaar))]
           < EXCLUDE_CENTERS_WITH_LESS_THAN_N_TAVIS_AFTER_FIRST_YEAR)) return(x)
    return(NULL)
  })
  try_log_info("Removing centers (%s), for having less than %d TAVIs in a year after their first year",
               paste0(remove_centers,collapse = ', '), EXCLUDE_CENTERS_WITH_LESS_THAN_N_TAVIS_AFTER_FIRST_YEAR)
  df1 = df1[!df1[, UOF_VN] %in% remove_centers,]
}
if(CAP_KREATININE_EGFR) {
  df1$kreatinine_gehalte = umap(df1$kreatinine_gehalte, function(x){ if(is.na(x)) NA else min(250,x)})
  df1$eGFR = umap(df1$eGFR, function(x){ if(is.na(x)) NA else min(120,x)})
}
if(LOG_TRANSFORM_3VARS) {
  df1$kreatinine_gehalte = log(df1$kreatinine_gehalte)
  df1$eGFR = log(df1$eGFR)
  df1$PA_druk = log(df1$PA_druk)
}
df1 = try_factorize_df_fl(df1)

##### main functions #############
# 2) model pipeline nested-CV [global model]:
run_nested_cv_pipeline <- function(df1, model_nm, var_selection_fn, model_fitting_fn, fold_splitting_fn, with_save,
                                   perform_lcoa = F, center_local_imputations = F, skip_folds = c()) {
  run_name_with_filepath = sprintf("../data/out/%s-%s.RData", model_nm ,"") #date()

  #   i) df1 split into k-folds, take k-1 of those and make them => df_train , remaining 1 => df_test
  df1$row_id = 1:nrow(df1)
  excluded.predictors = excluded.predictors[-which(excluded.predictors %in% grep(UOF_VN, excluded.predictors, value = T) )]
  kfolds = fold_splitting_fn(df1, N_FOLDS_CV, override.excluded.predictors = excluded.predictors)
  nested_cv_return_obj = NULL
  fitted_models = list()
  n_folds = len(kfolds$train_dfs)
  for(i in 1:n_folds){
    if(!RUN_HYPERPARAM_OPTIMIZATION){
      if(n_folds == 11) {
        try_log_info("Setting LR to %0.2f and aggreement_str_LASSO to %0.2f", FEDAVG_LR_10CV[i], FEDAVG_AGSLS_10CV[i])
        TRAINING_PARAMS_FL$learning_rate = FEDAVG_LR_10CV[i]
        TRAINING_PARAMS_FL$agrreement_strength_lasso = FEDAVG_AGSLS_10CV[i]
        TRAINING_PARAMS_FL <<- TRAINING_PARAMS_FL
      }
      if(n_folds == 17) {
        try_log_info("Setting LR to %0.2f and aggreement_str_LASSO to %0.2f", FEDAVG_LR_LCOA[i], FEDAVG_AGSLS_LCOA[i])
        TRAINING_PARAMS_FL$learning_rate = FEDAVG_LR_LCOA[i]
        TRAINING_PARAMS_FL$agrreement_strength_lasso = FEDAVG_AGSLS_LCOA[i]
        TRAINING_PARAMS_FL <<- TRAINING_PARAMS_FL
      }

    }
    if(i %in% skip_folds || is.null(kfolds$train_dfs[[i]]) || is.null(kfolds$test_dfs[[i]])) {
      try_log_debug(">>>>>>SKIP FOLD #%d", i)
      next
    }
    try_log_debug("FOLD #%d", i)
    HYPERPARAM_LAMBDA_FN <<- sprintf("lambd_fld=%d;mdl=%s;vs=%s.RData", i, model_nm, if(n_folds == N_FOLDS_CV+1) "CV" else "LCOA")
  # ii) impute separately one copy of df_train and df_test
    tmp = prep_df_train_test_imputation(kfolds$train_dfs[[i]], kfolds$test_dfs[[i]],
                                        center_local_imputations = center_local_imputations)

    if(len(tmp) == 0){
      try_log_warn("FOLD #%d - going to skip (failed to run prep_df_train_test_imputation...", i)
      next
    }
    test_row_ids = tmp$df_test$row_id
    train_row_ids = tmp$df_train$row_id
    # todo - clean this up
    df_train = tmp$df_train
    df_test = tmp$df_test
    df_train$row_id = NULL
    df_test$row_id = NULL

    tmp = prep_x_matricies(df_train, df_test)
    x_train = tmp$x_train
    y_train = df_train$outcome_of_interest

    x_train[,grep(UOF_VN, cns(x_train), val = T)] = NULL
    x_train[, UOF_VN] = as.numeric(levels(df_train[, UOF_VN]))[df_train[, UOF_VN]]

    x_test = tmp$x_test
    y_test = df_test$outcome_of_interest

    x_train = as.data.frame(x_train)
    x_train$outcome_of_interest = as.numeric(levels(df_train$outcome_of_interest))[df_train$outcome_of_interest]
    x_train = data.matrix(remove_columns_with_non_full_conf_matrix(x_train, remove_almostempty_quadrants = T ,
                                                                   cutoff_thresh = MIN_N_POS_OUTCOMES_IN_DATASET))
    x_test = as.data.frame(x_test)
    x_test$outcome_of_interest = df_test$outcome_of_interest

    #### ;;;;;;;;;;;;;;;;;;;;;;;;;; VARIABLE SELECTION ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    vars_selected = c()
    vars_selected = var_selection_fn(x_train, y_train)
    if(model_nm != 'fed-lr-lasso-gradient-update' && ONLY_VAR_SELECT)
      next


  # iv) using df_train & vars_selected fit a LR model
    #### ;;;;;;;;;;;;;;;;;;;;;;;;;; MODEL FITTING  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    c_lr_fit = model_fitting_fn(df_train, vars_selected, df_test)
    if(len(c_lr_fit) == 0) {
      try_log_warn("Empty model returned in fold %d", i)
      next
    }
    if(ONLY_VAR_SELECT)
      next
    fitted_models[[i]] = c_lr_fit
    #### ;;;;;;;;;;;;;;;;;;;;;;;;;; MODEL FITTING  END;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  # v) compute model predictions on df_test (and df_train if you want I guess...)
    lr_preds_train = try_predict(c_lr_fit, df_train)
    lr_preds_test = try_predict(c_lr_fit, df_test)
    vars_selected = if("glmnet" %in% class(vars_selected)) names(coef(vars_selected)[,1][which(coef(vars_selected)[,1] > 0) ]) else vars_selected

    if(any(lr_preds_train == -1)){
      idx_to_remove = which(lr_preds_train == -1)
      y_train = y_train[-idx_to_remove]
      lr_preds_train = lr_preds_train[-idx_to_remove]
    }
    if(any(lr_preds_test == -1)){
      idx_to_remove = which(lr_preds_test == -1)
      y_test = y_test[-idx_to_remove]
      lr_preds_test = lr_preds_test[-idx_to_remove]
      df_test = df_test[-idx_to_remove,]
      test_row_ids = test_row_ids[-idx_to_remove]
    }

    try_log_debug("##FOLD %d: LR train AUC = %0.3f", i,
                  try_get_aucroc_ci(lr_preds_train, as.numeric(y_train)-1)[2] )
    try_log_debug("##FOLD %d: LR test AUC = %0.3f", i,
                  try_get_aucroc_ci(lr_preds_test, as.numeric(y_test)-1)[2] )

    try_log_debug("##FOLD %d: LR train cali_int = %0.3f", i,
                  try_get_calibration(lr_preds_train, as.numeric(y_train)-1, verbose=F)$cal.intercept )
    try_log_debug("##FOLD %d: LR test cali_int = %0.3f", i,
                  try_get_calibration(lr_preds_test, as.numeric(y_test)-1, verbose=F)$cal.intercept )

    try_log_debug("##FOLD %d: LR train cali slp = %0.3f", i,
                  try_get_calibration(lr_preds_train, as.numeric(y_train)-1, verbose=F)$cal.slope )
    try_log_debug("##FOLD %d: LR test cali slp = %0.3f", i,
                  try_get_calibration(lr_preds_test, as.numeric(y_test)-1, verbose=F)$cal.slope )




    # vi)  save model,  predictions & vars selected
    #   df: {row_id}, {model-type}, {prediction}, {outcome}, {model-name} # model-name for linking with saved model
    c_row = as.data.frame(list(row_id = test_row_ids,
                               prediction = lr_preds_test,
                               outcome = as.numeric(y_test)-1,
                               center = df_test[, UOF_VN],
                               model_name = model_nm,
                               save_filepath = run_name_with_filepath,
                               fold = i
                               ))
    nested_cv_return_obj = rbind(nested_cv_return_obj, c_row)

  } # vii) repeat parts i) - vi) but on different test fold (total of 5 folds)

  # vi)  save model, test predictions & vars selected
  r_obj = list(preds_df = nested_cv_return_obj, fitted_models = fitted_models)
  if(with_save) {
    save_incremental_name(objToSave = r_obj,
                                      saveFilepath = run_name_with_filepath)
    try_log_info("Saved to %s", run_name_with_filepath)
  }
  return(r_obj)
}

#
# 3) pipeline bootstrapping (currently running with n_boot = 1, i.e. no bootstrap)
# i) random sample from df1 with repetition with same sample size => df_btstrp_smpl
# ii) run 2) [Nested CV] using df_brstrp_smpl

############ RUN CODE ##################
if(subsample_data != -1) {
  n_folds = round(1/subsample_data)
  splits = splitTools::create_folds(df1$outcome_of_interest, k = n_folds, type = "stratified")
  df1= df1[-splits$Fold1,]
}
if(RUN_CODE) {
  results_df = NULL
  df1_orig = df1
  orig_prev = howmany(df1_orig$outcome_of_interest == 1) / nrow(df1_orig)
  if(WITH_SINK)
    sink(SINK_NAME)
  c_prev = howmany(df1$outcome_of_interest == 1) / nrow(df1)
  if(RUN_GLOBAL) {
    try_log_debug("===============================> GLOBAL START  <===============================")
    model_nm = "lr-lasso-global"
    fold_splitting_fn = if(VALIDATION_STRAT == "LCOA") create_folds_lcoa else create_kfolds
    global_lasso = run_nested_cv_pipeline(df1, model_nm = model_nm,
                                                 var_selection_fn = try_LASSO_variable_selection_only,
                                                 model_fitting_fn = try_training_global_model_glm,
                                                 fold_splitting_fn = fold_splitting_fn,
                                                 with_save = SAVE_MODELS,
                                                 center_local_imputations = F,
                                                 skip_folds = SKIP_FOLDS)$preds_df

    a_row = as.data.frame(list(predictions = global_lasso$prediction,
                               outcomes = global_lasso$outcome, center = global_lasso$center))
    a_row$model = model_nm
    plot_calibration_graph(a_row$predictions, a_row$outcomes, smoothed = F, title = "Global LASSO")
  }
  if(RUN_LOCAL) {
    try_log_debug("===============================> LOCAL START  <===============================")
    if(VALIDATION_STRAT == "LCOA"){
      try_log_debug("===============================> LOCALS-ONLY  SKIP FOR LCOA  <===============================")
    }
    else {
      model_nm = "lr-lasso-locals"
      local_lasso = run_nested_cv_pipeline(df1, model_nm = model_nm,
                                              var_selection_fn = try_LASSO_local_models,
                                              model_fitting_fn = try_training_local_models_glm,
                                              fold_splitting_fn = create_kfold_strat_center,
                                              with_save = SAVE_MODELS,
                                              center_local_imputations = T,
                                              skip_folds = SKIP_FOLDS)$preds_df
      a_row = as.data.frame(list(predictions = local_lasso$prediction, outcomes = local_lasso$outcome,
                                 center = local_lasso$center))
      a_row$model = model_nm
    }
  }
  if(RUN_ENSEMBLE) {
    try_log_debug("===============================> ENSEMBLE LR FIT ONTOP  START  <===============================")
    model_nm = "fed-lr-lasso-ensemble"
    fold_splitting_fn = if(VALIDATION_STRAT == "LCOA") create_folds_lcoa else create_kfold_strat_center
    ensemble_lasso_lr_fit_ontop = run_nested_cv_pipeline(df1, model_nm = model_nm,
                                            var_selection_fn = try_LASSO_local_models,
                                            model_fitting_fn = try_fit_ensemble_and_lr_of_predictions,
                                            fold_splitting_fn = fold_splitting_fn,
                                            with_save = SAVE_MODELS,
                                            center_local_imputations = T,
                                            skip_folds = SKIP_FOLDS)
    ensemble_lasso_lr_fit_ontop = ensemble_lasso_lr_fit_ontop$preds_df
    a_row = as.data.frame(list(predictions = ensemble_lasso_lr_fit_ontop$prediction,
                               outcomes = ensemble_lasso_lr_fit_ontop$outcome,
                               center = ensemble_lasso_lr_fit_ontop$center))
    a_row$model = model_nm
  }
  if(RUN_FEDERATED) {
    try_log_debug("===============================> fedavg START  <===============================")

    model_nm = "fed-lr-lasso-gradient-update"
    fold_splitting_fn = if(VALIDATION_STRAT == "LCOA") create_folds_lcoa else create_kfold_strat_center
    federated_lasso = run_nested_cv_pipeline(df1, model_nm = model_nm,
                                             var_selection_fn = try_LASSO_dummy_allvars_returned, # var selection happens inside model_fitting_fn here
                                             model_fitting_fn = train_lr_federated_with_test_df,
                                             fold_splitting_fn = fold_splitting_fn,
                                             with_save = SAVE_MODELS,
                                             center_local_imputations = T,
                                             skip_folds = SKIP_FOLDS)$preds_df
    a_row = as.data.frame(list(predictions = federated_lasso$prediction, outcomes = federated_lasso$outcome,
                               center = federated_lasso$center))
    a_row$model = model_nm

  }
  if(WITH_SINK)
    sink()
}
