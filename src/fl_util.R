# Author: T.Y.
# Date created: 15.09.2022




################################################  plot functions   ################################################
plot_gd_lr_res <- function(lr_gd_res) {
  x_max = len(lr_gd_res$aucs_ci_ub)
  y_max = max(lr_gd_res$aucs_ci_ub)
  plot(1:x_max, lr_gd_res$loss, ylim = c(0, y_max), col = "white")


  converged_index = central_model$converged_index
  legend(x_max*0.7, y_max - y_max*0.8, legend = c("AUC ", "loss", sprintf("convergance (AUC = %0.2f)", central_model$aucs[converged_index])),
         col = c("green", "blue", "red"), lty = c(1,1,2), lwd = c(2, 2, 1.5) ,
         bg = "lightblue")

  lines(1:x_max, lr_gd_res$loss, ylim = c(0, y_max), col = "blue", lwd = 2)
  lines(1:x_max, lr_gd_res$aucs_ci_ub, col = "orange", lwd = 1.5)
  lines(1:x_max, lr_gd_res$aucs_ci_lb, col = "orange", lwd = 1.5)
  lines(1:x_max, lr_gd_res$aucs, col = "green", lwd = 2)
  lines(c(converged_index, converged_index) , c(0, y_max), col = "red", lty = 2, lwd = 1.5 )
  grid(16,16)



  if(converged_index > 0)
    try_log_debug("CENTRAL MODEL, AUC on convergence point (iter #%d) = %0.2f [95%%CI %0.2f - %0.2f];", converged_index,
                  central_model$aucs[converged_index], central_model$aucs_ci_lb[converged_index],
                  central_model$aucs_ci_ub[converged_index])
}



################################################  maths functions   ####################################

sigmoid <- function(x) { return( 1/(1+exp(-x)) ) }


count_decimal_digits <- function(x) {
  x_str <- format(x, scientific = FALSE)  # Convert to character without scientific notation
  decimal_part <- substring(x_str, first = regexpr("\\.", x_str) + 1)  # Extract decimal part
  non_zero_position <- regexpr("[1-9]", decimal_part)  # Find position of first non-zero digit
  if (non_zero_position == -1) {
    return(0)  # No non-zero digits after decimal point
  } else {
    # digits_after_decimal <- nchar(decimal_part) - non_zero_position + 1
    return(as.numeric(non_zero_position))
  }
}

################################################  predict functions ################################################

try_predict <- function(m1, newdata) {
  newdata_matrix = prep_x_matrix(newdata, do_remove_columns_with_non_full_conf_matrix = F)
  if(FEDERATION_MECHANISM_UPDATE_SHARING_LIST %in% class(m1))
    return(PRVT__try_predict_model_updates_list(m1, newdata_matrix))
  if(FEDERATION_MECHANISM_UPDATE_SHARING %in% class(m1))
    return(PRVT__try_predict_model_updates(m1$coefficients, newdata_matrix, m1$recal_fn))
  if(FEDERATION_MECHANISM_ENSEMBLE_OF_MODELS %in% class(m1))
    return(PRVT__try_predict_ensemble_of_models(m1, as.data.frame(newdata_matrix)))
  if(FEDERATION_MECHANISM_ENSEMBLE_OF_MODELS_WITH_LR_FIT_ON_PREDICTIONS %in% class(m1))
    return(PRVT__try_predict_ensemble_of_models_with_lr_fit_on_predictions(m1, as.data.frame(newdata_matrix)))
  if(FEDERATION_MECHANISM_LOCAL_MODELS_ONLY %in% class(m1))
    return(PRVT__try_predict_local_models_only(m1, as.data.frame(newdata_matrix)))
  if("glmnet" %in% class(m1))
    return(predict_glmnet(m1, newdata_matrix))
  if("glm" %in% class(m1))
    return(predict(m1, as.data.frame(newdata_matrix), type = "response"))
  else
    return(logit2prob(predict(m1, as.data.frame(newdata_matrix))))
}

predict_glmnet <- function(m1, x) {
  x = as.data.frame(x)
  coef(m1)[,m1$lambda2use_idx]
  x = x[,names(coef(m1)[,m1$lambda2use_idx])[-1]]
  x = data.matrix(x)
  return(as.vector(logit2prob(predict(m1, x)[,m1$lambda2use_idx])))
}

try_get_preds_df_ensemble <- function(lasso_models_list, x_train, y_train = NULL, is_glm = T) {
  preds_df = NULL
  c_names = c()
  for(i in 1:len(lasso_models_list)) {
    c_m = lasso_models_list[[i]]
    c_names = c(c_names, last(class(c_m)))
    c_preds = c()
    if(is_glm) {
      x_train = as.data.frame(x_train)
      cols_missing  = setdiff(names(c_m[[1]]$coefficients), cns(x_train))
      if(len(cols_missing) > 0)
        x_train[, cols_missing] = 0
      c_preds = predict(c_m[[1]], x_train , type="response")
    }
    else
      c_preds = predict_glmnet(c_m, x_train)
    preds_df = cbind(preds_df, c_preds)
  }
  preds_df = as.data.frame(preds_df)
  colnames(preds_df) = c_names
  preds_df$outcome_of_interest = y_train # so here you are sharing the outcome between partitions
  return(preds_df)
}

PRVT__try_predict_model_updates_list <- function(l_models, x){
  c_preds = c()
  for(i in 1:len(l_models)){
    c_m = l_models[[i]]
    if(i == 1)
      c_preds = PRVT__try_predict_model_updates(c_m$coefficients, x, c_m$recal_fn)
    else
      c_preds = c_preds + PRVT__try_predict_model_updates(c_m$coefficients, x, c_m$recal_fn)
  }
  return(c_preds/len(l_models))
}
PRVT__try_predict_model_updates <- function(betas, x, recal_fn = NULL, get_logits = F) {

  if(len(betas) > 2) {
    missing_from_x = setdiff(names(betas)[-1], cns(x))
    if(len(missing_from_x) > 0){
      x = as.data.frame(x)
      x[,missing_from_x] = 0
      x = as.matrix(x)
    }
    x = x[,names(betas)[-1]]
  }
  else { # x should be a column vector if its just one variable selected, but here it arrives as a vector
    if(all(class(x) == "numeric"))
      x = as.data.frame(list(a = x))
    else {
      x = x[,names(betas)[-1]]
      x = as.data.frame(list(a = x))
    }
    colnames(x) = names(betas)[2]
  }

  x = add_intercept_col_and_convert_to_matrix(x)
  #RULE for left %*% right : ncol(left) == nrow(right)
  res = as.vector ( t( x %*% betas)  )  # also works (betas %*% t(x))
  res_probs = sigmoid(res)
  if(!is.null(recal_fn) && FEDAVG_PERFORM_L1_RECALL) {
    res_probs = recal_fn(res_probs)
    res = boot::inv.logit(res_probs)
  }

  if(get_logits)
    return(res)
  else
    return(res_probs) # remember : inv.logit === sigmoid , i.e. sigmoid === boot::inv.logit
}

PRVT__try_predict_local_models_only <- function(m1, x) {
  preds = NULL
  x$row_id = 1:nrow(x)
  x = try_add_removed_category_from_ohe(x, grep_on_colnames(x, UOF_VN), sprintf("%s1", UOF_VN)) # because ohe removes reduntant cats , e.g. [1,2,3] -> [2,3]
  x = try_place_column_in_order_df(x, cols_to_remove = c(), col_to_add =  x[, sprintf("%s1", UOF_VN)],
                                   col_nm  = sprintf("%s1", UOF_VN), col_pos =  which(cns(x) == sprintf("%s2", UOF_VN)))
  x[,sprintf("%s1.1", UOF_VN)] = NULL
  x = try_convert_ohe_categorical_to_cont(x, grep_on_colnames(x, UOF_VN), cont_var_nm = UOF_VN)
  x[, UOF_VN] = x[, UOF_VN] + 1

  for(c_cent in sort(uniq(x[, UOF_VN]))){
    c_df = x[x[, UOF_VN] == c_cent,]  # get center df
    c_model = map(1:len(m1),  function(i) { #lookup corresponding model for center df
      c_center_class = class(m1[[i]])[grep(UOF_VN, class(m1[[i]]))]
      c_center_num = as.numeric(substr(c_center_class, 8, strlen(c_center_class)))
      if(c_center_num == c_cent)
        return(m1[[i]])
      else
        return(NULL)
    })
    match_found =  F
      if(len(c_model) != 0)
      for(i in 1:len(c_model)){
        if(len(c_model[[i]]) != 0 && !match_found) {
          c_model = c_model[[i]]
          match_found = T
          break
        }
      }

    if(match_found) {
      c_glm = c_model[[1]]
      c_preds = as.data.frame( list( preds = predict(c_glm, c_df, type = "response") , row_ids = c_df$row_id))
      preds = rbind(preds, c_preds)
    }
    else {
      c_preds = as.data.frame( list( preds = rep(-1, nrow(c_df)) , row_ids = c_df$row_id))
      preds = rbind(preds, c_preds)
    }
  }
  # preds = preds / len(m1)
  preds = preds[order(preds$row_id),]

  return(preds$preds)
}

PRVT__try_predict_ensemble_of_models <- function(m1, x) {
# todo: rework logic here, m1 is now a list of models of glmnet
  preds = c()
  for(i in 1:len(m1)){
    preds = if(i == 1) predict_glmnet(m1[[i]], x) else preds + predict_glmnet(m1[[i]], x)
  }
  preds = preds / len(m1)
  return(preds)
}

PRVT__try_predict_ensemble_of_models_with_lr_fit_on_predictions <- function(m1, x) {
  lasso_models_list = m1$lasso_models_list
  lr_fit = m1$lr_from_local_models
  recal_fn = m1$recal_fn
  preds_df = try_get_preds_df_ensemble(lasso_models_list, x)
  c_names = names(lr_fit$coefficients)
  lr_fit_preds = c()

  # if no L2 recalibration was performed, just get the mean volume-weighted average, same as fedavg
  # thecn check if L1 recalibration was performed and apply if so
  if(is.null(lr_fit)) {
    predictor_idxs = grep(sprintf("%s_", UOF_VN), cns(preds_df))
    center_weights = m1$center_weights
    for(c_row in 1:nrow(preds_df)) {
      c_dfr = preds_df[c_row,predictor_idxs]
      c_pred = mean(t(t(c_dfr) * as.data.frame(center_weights))  )
      c_pred = mean(t(c_dfr))
      lr_fit_preds = c(lr_fit_preds, c_pred)
    }
    # check if l1 recal was performed, apply it here if so
    if(!is.null(recal_fn)){
      lr_fit_preds = recal_fn(lr_fit_preds)
    }
  }
  # by here we know L2 recalibration was performed (as lr_fit is not null)
  else {
    lr_fit$coefficients = umap(lr_fit$coefficients, function(x) { if(is.na(x)) 0 else x })
    names(lr_fit$coefficients) = c_names
    lr_fit_preds = PRVT__try_predict_model_updates(lr_fit$coefficients, data.matrix(preds_df))
  }
  return(lr_fit_preds)
}

################################################  metrics / loss functions ################################################

sqrt_err <- function(betas, x, y, class_weighting = F) {
  preds = PRVT__try_predict_model_updates(betas, x)
  sq_errs = (y - preds ) ^ 2
  if(class_weighting) {
    y1_prev = howmany(y == 1) / len(y)
    y1_idxs = which(y == 1)
    sq_errs[y1_idxs] =  sq_errs[y1_idxs] *  (2-y1_prev)
    sq_errs[-y1_idxs] =  sq_errs[-y1_idxs] *  (1+ y1_prev)
  }
  return(sq_errs)
}

mse <- function(betas, x, y, class_weighting = F) {
  return(mean(sqrt_err(betas, x, y, class_weighting)))
}

inv_auc <- function(betas, x, y) {
  preds = PRVT__try_predict_model_updates(betas, x)
  return(1-try_get_aucroc(predictions = preds, true_vals =  y))
}

mse_inv_auc_combo <- function(betas, x, y, weight_mse = 0.5) {
  return(mse(betas,x,y)*weight_mse+inv_auc(betas,x,y)*(1-weight_mse))
}

try_logloss <- function(y, preds) {
  logloss = mean(( y * log(preds)  + (1-y)*log(1-preds) ) * (-1)  )
  return(logloss)
}

try_mse <- function(y, preds) {
  return(mean((y - preds)^2))
}

try_cost_auc <- function(y, preds) {
  try_get_aucroc_ci(preds, y)
}

################################################  util / transform functions ################################################

feature_scale <- function(x) {
  min_val <- min(x)
  max_val <- max(x)
  scaled <- (x - min_val) / (max_val - min_val)
  return(scaled)
}

remove_non_imputable_columns <- function(df1) {
  col_to_remove <- c()
  for(cn in cns(df1)){
    print(cn)
    print(table(df1[,cn], df1$outcome_of_interest))
    if(len(uniq(df1[,cn])) > MAX_N_CATEGORIES_IN_DATASET || cn == "outcome_of_interest")
      next

    if(any(table(df1[,cn], df1$outcome_of_interest) <= 8) || nrow(table(df1[,cn])) < 2)
      col_to_remove <- c(col_to_remove, cn)
  }
  try_log_info("@remove_non_imputable_columns- removing following columns (reason: non-NA vals seen only when outcome_of_interest = 0 or 1): \n[ %s ];",
               paste(col_to_remove, collapse = ", "))
  df1 <- df1[,setdiff(cns(df1),col_to_remove)]
  df1
}

remove_columns_with_non_full_conf_matrix <- function(df1, remove_almostempty_quadrants = F, cutoff_thresh = 10) {
  if(remove_almostempty_quadrants) {
    clsms_to_remove = try_table_per_column(df1, warn_thresh = cutoff_thresh, quiet = T)
    cols_to_exclude = clsms_to_remove[grep(sprintf("(%s*|outcome_of_interest)", UOF_VN) ,clsms_to_remove)]
    clsms_to_remove = setdiff(clsms_to_remove, cols_to_exclude)
    df1[,clsms_to_remove] = NULL
    return(df1)
  }
  return(remove_non_imputable_columns(df1))
}

add_intercept_col_and_convert_to_matrix <- function(x) {
  n_cns = c("[Intercept]", cns(x))
  x = cbind(rep(1, nrow(x)), x)
  colnames(x) = n_cns
  return(as.matrix(x))
}

autofill_missing_var_columns <- function(df1, vars, autofill_value = 0, verbose = F) {
  for(col_nm in setdiff(vars, cns(df1))) {
    df1[,col_nm] = autofill_value
    if(verbose)
      try_log_debug("@autofill_missing_var_columns - called for column %s", col_nm)
  }
  df1
}

partition_df_from_pipeline <- function(df1) {
  out_df_list = list()
  for(c_fed in sort(uniq(df1[, UOF_VN]))) {
    out_df_list[[c_fed]] = df1[df1[, UOF_VN] == c_fed,]
  }
  names(out_df_list) = 1:len(out_df_list)
  return(out_df_list)
}

prep_x_matricies <- function(df_train, df_test) {
  x_train = as.data.frame(prep_x_matrix(df_train, do_remove_columns_with_non_full_conf_matrix = T))
  x_test = as.data.frame(prep_x_matrix(df_test, do_remove_columns_with_non_full_conf_matrix = F))

  # y_train = df_train$outcome_of_interest
  # y_test = df_test$outcome_of_interest

  # x_train$outcome_of_interest = y_train
  # x_test$outcome_of_interest = y_test
  # tmp = remove_columns_not_present_in_both_dfs(x_train, x_test) # todo: see if we can skip this
  # tmp$df_train$outcome_of_interest = NULL
  # tmp$df_test$outcome_of_interest = NULL
  # return(list(x_train= tmp$df_train, x_test = tmp$df_test))
  x_train$outcome_of_interest = NULL
  x_test$outcome_of_interest = NULL
  return(list(x_train= x_train, x_test = x_test))
}

prep_x_matrix <- function(df1, do_remove_columns_with_non_full_conf_matrix, remove_almostempty_quadrants = T) {
  for(c_col in cns(df1)) {
    na_idxs = which(is.na(df1[,c_col]))
    if(len(na_idxs) != 0) {
      is_factor = is.factor(df1[na_idxs,c_col])
      if(is_factor){
        a = addNA(df1[na_idxs,c_col])
        levels(df1[,c_col]) = c(levels(df1[,c_col]), NA_PLACEHOLDER)
        levels(a) = levels(df1[,c_col])
        df1[na_idxs,c_col] = a
      }
      else
        df1[na_idxs,c_col] = NA_PLACEHOLDER
    }
  }
  x = model.matrix(outcome_of_interest~., df1)[,-1]
  x =  as.data.frame(x)
  x$outcome_of_interest = df1$outcome_of_interest
  if(do_remove_columns_with_non_full_conf_matrix)
    x = remove_columns_with_non_full_conf_matrix(x, remove_almostempty_quadrants = remove_almostempty_quadrants)
  x$outcome_of_interest = NULL
  x = data.matrix(x)
  return(x)
}

prep_df_train_test_imputation <- function(df_train, df_test, n_copies = 1, center_local_imputations = F) {
  orig_seed = random.seed
  random.seed = sample(1:1000000,1)
  imp_df_train = NULL
  imp_df_test = NULL
  if(n_copies != 1 && center_local_imputations){
    try_log_error("@prep_df_train_test_imputation n_copies != 1 && center_local_imputations=T - not implemented for more than one copies")
    exit(-1)
  }
  is_lcoa = len(uniq(df_test[, UOF_VN]) ) == 1
  test_imputed_lcoa = F
  if(center_local_imputations) {
    for(i in sort(uniq(df_train[, UOF_VN]))){ # i = sort(uniq(df_train[, UOF_VN]))[1]
      c_df_train = df_train[df_train[, UOF_VN] == i,]
      c_df_test = if(is_lcoa) df_test else df_test[df_test[, UOF_VN] == i,]

      c_df_train_imp = impute_n_copies(c_df_train, n_copies = 1)[[1]]
      # Potential issue -  when its 10-CV c_df_test_imp does not get imputed !
      # fixed now - try re-runing?
      c_df_test_imp = NULL
      if(is_lcoa){
        if(!test_imputed_lcoa)
          c_df_test_imp = impute_n_copies(c_df_test, n_copies = 1)[[1]]
        else
          c_df_test_imp = c_df_test
      }
      else
        c_df_test_imp = impute_n_copies(c_df_test, n_copies = 1)[[1]]

      if(len(c_df_train_imp) == 0 || len(c_df_test_imp) == 0)
        return(NULL)

      c_df_train_imp[, UOF_VN] = c_df_train[, UOF_VN]
      c_df_test_imp[, UOF_VN] = c_df_test[, UOF_VN]

      cols_missing_train = setdiff( cns(df_train), cns(c_df_train_imp))
      c_df_train_imp[, cols_missing_train] = NA # set removed/unimputable cols to NAs
      c_df_train_imp = c_df_train_imp[,  cns(df_train) ] # keep same order

      cols_missing_test = setdiff( cns(df_test), cns(c_df_test_imp))
      c_df_test_imp[, cols_missing_test] = NA # set removed/unimputable cols to NAs
      c_df_test_imp = c_df_test_imp[,  cns(df_test) ] # keep same order

      # (all (cns(df_train) == cns(c_df_train_imp)))

      df_train[df_train[, UOF_VN] == i,] = c_df_train_imp
      if(is_lcoa){
        if(!test_imputed_lcoa){
          df_test = c_df_test_imp
          test_imputed_lcoa = T
        }
      }
      else
        df_test[df_test[, UOF_VN] == i,] = c_df_test_imp
    }
    # df_train[, UOF_VN] = NULL
    # df_test[, UOF_VN] = NULL
    df_train = list(df_train)
    df_test = list(df_test)
  }
  else{
     df_train = impute_n_copies(df_train, n_copies)
     df_test = impute_n_copies(df_test, n_copies)
   }

  random.seed = orig_seed
  for(i in 1:n_copies){
    tmp = remove_columns_not_present_in_both_dfs(df_train[[i]], df_test[[i]])
    df_train[[i]] = tmp$df_train
    df_test[[i]] = tmp$df_test
  }
  if(n_copies == 1){
    df_train = df_train[[1]]
    df_test = df_test[[1]]
  }
  return(list(df_train = df_train, df_test = df_test))
}

remove_columns_not_present_in_both_dfs <- function(df_train, df_test){
  cns_test = cns(df_test)
  cns_train = cns(df_train)
  present_in_train_missing_in_test = setdiff(cns_train, cns_test)
  present_in_test_missing_in_train = setdiff(cns_test, cns_train)
  if(len(present_in_train_missing_in_test) > 0){
    df_train[,present_in_train_missing_in_test] = NULL
  }
  if(len(present_in_test_missing_in_train) > 0){
    df_test[,present_in_test_missing_in_train] = NULL
  }
  return(list(df_train = df_train, df_test = df_test))
}

build_res_df_for_extract_results_df <- function(c_df, c_cent, stratify_on) {
  c_auc = c(NA,NA,NA) #NULL
  c_cali = NULL
  # auc
  tryCatch(
    expr = {
      c_auc = try_get_aucroc_ci(c_df$prediction, c_df$outcome)
    },
    error = function(e){
      c_auc= c(NA,NA,NA)
      try_log_error("failed to get auc")
    },
    warning = function(w){
      # (Optional)
      # Do this if an warning is caught...
    },
    finally = {
      # (Optional)
      # Do this at the end before quitting the tryCatch structure...
    }
  )
  #cali
  tryCatch(
    expr = {
      c_cali = try_get_calibration(p = c_df$prediction, y = c_df$outcome, verbose = F)
    },
    error = function(e){
      try_log_error("failed to get calibration metrics")
    },
    warning = function(w){
      # (Optional)
      # Do this if an warning is caught...
    },
    finally = {
      # (Optional)
      # Do this at the end before quitting the tryCatch structure...
    }
  )
  plot_nm = "none"
  n_row = as.data.frame(list(model_nm = c_df$model_name[1],
                             auc = c_auc[2],
                             auc_95lo = c_auc[1],
                             auc_95_hi = c_auc[3],
                             cali_int = if(len(c_cali$cal.intercept) == 0) NA else c_cali$cal.intercept,
                             cali_int_95lo = if(len(c_cali$ci.cal.intercept[1]) == 0) NA else c_cali$ci.cal.intercept[1],
                             cali_int_95hi = if(len(c_cali$ci.cal.intercept[2]) == 0) NA else c_cali$ci.cal.intercept[2],
                             cali_slp = if(len(c_cali$cal.slope) == 0) NA else c_cali$cal.slope,
                             cali_slp_95lo = if(len(c_cali$ci.cal.slope[1]) == 0) NA else c_cali$ci.cal.slope[1],
                             cali_slp_95hi = if(len(c_cali$ci.cal.slope[2]) == 0) NA else c_cali$ci.cal.slope[2],
                             cali_graph_file = plot_nm
  ))
  n_row[,stratify_on] = c_cent
  return(n_row)
}

extract_results_df <- function(pipeline_res_df, plot_cali_graph = T, stratify_on = "center") {
  pipeline_res_df$center = as.numeric(levels(pipeline_res_df$center)[pipeline_res_df$center]) # ensure factors are translated well
  # validate config
  if(len(stratify_on) != 1 &&
     (len(stratify_on) != 2 && all( stratify_on  == c('fold', 'center')) ) &&
     all(stratify_on %in% c("center", "fold")))
  {
    try_log_error("@extract_results_df - Invalid value for stratify_on. must be oneOf :[ fold, center, or c(fold, center) (for local model)  ]")
    quit(-1)
  }
  if(len(pipeline_res_df) == 0)
    return(NULL)

  res_df = NULL
  for(c_cent in c('ALL',sort(uniq(pipeline_res_df[,stratify_on[1]]))) ) {
    # print(c_cent)
    c_df = pipeline_res_df
    if(c_cent != 'ALL')
      c_df = pipeline_res_df[pipeline_res_df[,stratify_on[1]] == c_cent,]
    if(nrow(c_df) < 10 || howmany(c_df$outcome == 1)  <= 1)
      next

    if(len(stratify_on) == 2) { # only for local model which is CV, so it will be [1]:= per $fold{ [2]per $center  { .. } }
      inner_res_df = NULL
      for(c_cent4r in sort(uniq(c_df[,stratify_on[2]]))){
        cc_df = c_df[c_df[,stratify_on[2]] == c_cent4r,]
        # try_log_debug("%s = %s; %s = %s (n=%d, c=%d)", stratify_on[1], c_cent,
        #               stratify_on[2], c_cent4r, nrow(cc_df), howmany(cc_df$outcome==1))
        c_res = build_res_df_for_extract_results_df(c_df = cc_df, c_cent = c_cent4r, stratify_on = stratify_on[2])
        inner_res_df = rbind(inner_res_df, c_res)
      }
      inner_res_df[, stratify_on[1]] = c_cent
      res_df = rbind(res_df, inner_res_df)
    }
    else { # else we just stratify on the one thing specified
      plot_nm = "none"
      if(plot_cali_graph){
        plot_nm = sprintf("cali_graph_%s_c%s.bmp", uniq(c_df$model_name),c_cent)
        plot_filepath = sprintf("%s\\%s", "G:\\divjk\\kik\\NHR-Onderzoek\\Abu Hanna\\R scripts(TY)\\PhD\\FL\\plots\\cali",
                                plot_nm)

        bmp(file=plot_filepath, width=32, height=16, pointsize = 22, units = 'in', res = 170)
        # art
        # plot.new()
        # for(i in 1:200) { abline(0, 1, col = colors()[( (i*3-round(sqrt(i)))%%len(colors()) )+ 1], lwd = 2000-i*10)
        #                   abline(1, -1, col = colors()[( (i*2+ round(log(i)))%%len(colors()) )+ 1], lwd = 2000-i*10)
        #                 }
        par(mfrow=c(1,2))
        plot_calibration_graph(c_df$prediction, c_df$outcome, smoothed = F, add_quantiles_and_mean_text = T,
                               excluded_percentiles = c(000,100), title = sprintf("%s", plot_nm))
        plot_calibration_graph(c_df$prediction, c_df$outcome, smoothed = F, add_quantiles_and_mean_text = T,
                               excluded_percentiles = c(000,95), title = "top 5% predictions excluded")
        dev.off()
      }
      c_res = build_res_df_for_extract_results_df(c_df = c_df, c_cent = c_cent, stratify_on = stratify_on[1])
      res_df = rbind(res_df, c_res)

    }
  }
  # compute 'ALL' fold/center , simple mean , when stratify on 1 thing only
  if(len(stratify_on) == 1) {
    if(len(res_df)!=0 ) {
      if(nrow(res_df) > 0 && !(nrow(res_df)==1 &&  res_df$fold == 'ALL')) {
        m_row = as.data.frame(list(model_nm = c_df$model_name[1],
                                   auc = mean(res_df$auc[-1], na.rm = T),
                                   auc_95lo = mean(res_df$auc_95lo[-1], na.rm = T),
                                   auc_95_hi = mean(res_df$auc_95_hi[-1], na.rm = T),
                                   cali_int = mean(res_df$cali_int[-1], na.rm = T),
                                   cali_int_95lo = mean(res_df$cali_int_95lo[-1], na.rm = T),
                                   cali_int_95hi = mean(res_df$cali_int_95hi[-1], na.rm = T),
                                   cali_slp = mean(res_df$cali_slp[-1], na.rm = T),
                                   cali_slp_95lo = mean(res_df$cali_slp_95lo[-1], na.rm = T),
                                   cali_slp_95hi = mean(res_df$cali_slp_95hi[-1], na.rm = T),
                                   cali_graph_file = 'no'
        ))
        m_row[,stratify_on] = 'mean'
        res_df= rbind(res_df, m_row)
      }
    }
  }
  # here pool first per center, over all folds . Then rema out of that.
  if(len(stratify_on) == 2) {
    for(c_cent in sort(uniq(res_df$center))) {
      c_res = res_df[res_df$center == c_cent,]
      c_res = c_res[!c_res$fold %in% c('ALL', 'mean'),]
      non_na_aucs = which(!is.na(c_res$auc))
      non_na_calis = which(!is.na(c_res$cali_int))
      n_res_auc = len(non_na_aucs)
      n_res_cali = len(non_na_calis)
      n_row = c_res[1,]
      n_row$fold = 'mean'
      # pooled_mean_ci_low = mean_of_means - (sd(c_res$auc, na.rm=T)/sqrt(howmany(!is.na(c_res$auc))))*1.96,
      # pooled_mean_ci_hi = mean_of_means + (sd(c_res$auc, na.rm=T)/sqrt(howmany(!is.na(c_res$auc))))*1.96,
      n_row$auc = mean(c_res$auc, na.rm = T)
      if(n_res_auc > 1) {
        n_row$auc_95lo = n_row$auc - (sd(c_res$auc, na.rm=T)/sqrt(howmany(!is.na(c_res$auc))))*1.96
        n_row$auc_95_hi = n_row$auc + (sd(c_res$auc, na.rm=T)/sqrt(howmany(!is.na(c_res$auc))))*1.96
      }  # otherwise just keep original 95ci


      n_row$cali_int = mean(c_res$cali_int, na.rm = T)
      if(n_res_cali > 1) {
        n_row$cali_int_95lo = n_row$cali_int - (sd(c_res$cali_int, na.rm=T)/sqrt(howmany(!is.na(c_res$cali_int))))*1.96
        n_row$cali_int_95hi = n_row$cali_int + (sd(c_res$cali_int, na.rm=T)/sqrt(howmany(!is.na(c_res$cali_int))))*1.96
      }  # otherwise just keep original 95ci

      n_row$cali_slp = mean(c_res$cali_slp, na.rm = T)
      if(n_res_cali > 1) {
      n_row$cali_slp_95lo = n_row$cali_slp - (sd(c_res$cali_slp, na.rm=T)/sqrt(howmany(!is.na(c_res$cali_slp))))*1.96
      n_row$cali_slp_95hi = n_row$cali_slp + (sd(c_res$cali_slp, na.rm=T)/sqrt(howmany(!is.na(c_res$cali_slp))))*1.96
      }  # otherwise just keep original 95ci
      res_df = rbind(res_df, n_row)
    }
  }
  return(res_df)
}

try_bootstrap_sample <- function(x, nboot = 2000){
  all_sampls = NULL
  for(i in 1:nboot){
    c_smpl = sample(x, replace = T, size = len(x))
    names(c_smpl) = sprintf('b%d',1:len(x))
    c_smpl = t(c_smpl)
    all_sampls = rbind(all_sampls, c_smpl)
  }

  return(all_sampls)
}

try_factorize_df_fl <- function(df1, only_bin = F) {
  tmp = try_print_class_for_each_column_df(df1, quiet = T)
  should_be_categorical_vars = tmp$should_be_categoricals
  should_be_bin_vars = tmp$should_be_binaries


  for(bin_var in should_be_bin_vars) {
    c_v = df1[,bin_var]
    df1[,bin_var] =  try_factorize_col(c_v)
  }
  if(!only_bin) {

    should_be_categorical_vars
    cat_vars_groupings = list(diabetes = list(DM_n = c(0), DM_y = c(1,2,10,20,30,90)),
                              NYHA = list(),
                              urgentie = list(elective = c(10), urgent = c(20), emergency = c(30,40)),
                              TAVI_toegang = list(other_none = c(0,90), transfemoral = c(10,11,12), subclavian = c(15), transapical = c(25), direct_aortic = c(30)),
                              interv_gewicht = list(other = c(10,11,30), two_operations = c(20)),
                              jaar = list()
    )
    cat_vars_groupings[[UOF_VN]] = list()

    for(i in 1:len(cat_vars_groupings)){
      cat_var = names(cat_vars_groupings)[i]
      cat_var_grouping = cat_vars_groupings[[i]]
      c_v = df1[,cat_var]

      df1[,cat_var] = try_factorize_col(c_v, cat_var_grouping)
    }
  }
  return(df1)
}

################################################  util / string manipulation  ################################################

transform_strings_via_mapping <- function(strs, mapping, do_print = T) {
  mapped_strs = umap(strs, function(x) { if(x %in% names(mapping)) mapping[[x]] else x})
  if(do_print)
    cat(sprintf("%s\n", mapped_strs))
  return(mapped_strs)
}


################################################  train model functions ################################################


# @return ::
#      - vector containing coefficients of trained model ( when federation_mechanism = "model-update-transmission" )
#      - dataframe with columns "model_lab", "coefficient_nm", "coefficient_val" ( when federation_mechanism = ""ensemble-of-models"" )
# @aggr_technique  = c("average", "fedavg") # only used when federation_mechanism = "model-update-transmission"
# @federation_mechanism = c("model-update-transmission", "ensemble-of-models")
# @do_feature_selection_per_partition - only valid for  federation_mechanism = "ensemble-of-models"
train_federated_logreg <- function(fed_dfl, federation_mechanism, training_params) {

  fl_fn_mapper = list("model-update-transmission" = PRVT__train_federated_logreg_model_update,
                      "ensemble-of-models" = try_bootstrap_kfold_fl_ensemble)

  return(fl_fn_mapper[[federation_mechanism]](fed_dfl, training_params))

}

train_lr_global <- function(df_train, vars_selected) {
  # centr_col = df_train[, UOF_VN]
  df_train[, UOF_VN] = NULL
  matrix_train = prep_x_matrix(df_train, do_remove_columns_with_non_full_conf_matrix = F)
  matrix_train = as.data.frame(matrix_train)
  matrix_train$outcome_of_interest = df_train$outcome_of_interest
  # matrix_train[, UOF_VN] = centr_col
  matrix_train = as.data.frame(data.matrix(matrix_train))
  matrix_train$outcome_of_interest =   matrix_train$outcome_of_interest -1
  return(train_and_pool_logreg(list(matrix_train), vars_selected, with_pooling = F)[[1]])
}

train_lr_federated <- function(df_train, vars_selected, test_df = NULL) {
  if(len(test_df) == 0)
    test_df = df_train
  matrix_test = prep_x_matrix(test_df, do_remove_columns_with_non_full_conf_matrix = F)
  matrix_test =  as.data.frame(matrix_test)
  matrix_test$outcome_of_interest = test_df$outcome_of_interest
  matrix_test[, UOF_VN] = test_df[, UOF_VN]
  matrix_test = as.data.frame(data.matrix(matrix_test))
  matrix_test$outcome_of_interest =  matrix_test$outcome_of_interest -1
  train_params = TRAINING_PARAMS_FL
  train_params$test_df  = matrix_test
  train_params$vars_selected = vars_selected

  centr_col = df_train[, UOF_VN]
  df_train[, UOF_VN] = NULL
  matrix_train = prep_x_matrix(df_train, do_remove_columns_with_non_full_conf_matrix = F)
  matrix_train =  as.data.frame(matrix_train)
  matrix_train$outcome_of_interest = df_train$outcome_of_interest
  matrix_train[, UOF_VN] = centr_col
  matrix_train = as.data.frame(data.matrix(matrix_train))
  matrix_train$outcome_of_interest =   matrix_train$outcome_of_interest -1
  paritioned_df_train = partition_df_from_pipeline(matrix_train)
  res = train_federated_logreg(paritioned_df_train, federation_mechanism  = "model-update-transmission", train_params)

  return(res)
}

train_lr_ensemble <- function(df_train, vars_selected) {
  training_params = list(vars_selected = vars_selected,
                         test_df = df_train,
                         weighting_scheme = "mean",
                         do_feature_selection_per_partition = F,
                         with_cv = F,
                         cv_n_folds = 1,
                         cv_with_perf_metrics = F,
                         with_bootstrap = F,
                         bootstrap_n_inter = 1,
                         verbose = T)

  centr_col = df_train[, UOF_VN]
  df_train[, UOF_VN] = NULL
  matrix_train = prep_x_matrix(df_train, do_remove_columns_with_non_full_conf_matrix = F)
  matrix_train =  as.data.frame(matrix_train)
  matrix_train$outcome_of_interest = df_train$outcome_of_interest
  matrix_train[, UOF_VN] = centr_col
  matrix_train = as.data.frame(data.matrix(matrix_train))
  matrix_train$outcome_of_interest =   matrix_train$outcome_of_interest -1
  paritioned_df_train = partition_df_from_pipeline(matrix_train)

  res = PRVT__train_federated_logreg_ensemble(paritioned_df_train, training_params)

  # res = train_federated_logreg(paritioned_df_train, federation_mechanism = FEDERATION_MECHANISM_ENSEMBLE_OF_MODELS,
  # training_params = training_params)
  return(res)
}

try_bootstrap_kfold_fl_ensemble <- function(fed_dfl, training_params = NULL) {
  # init params
  list2env(training_params, envir = environment())
  try_log_debug("@try_bootstrap_kfold_fl_ensemble - called with following params:")
  cat(str(training_params))

  preds = NULL
  true_vals = NULL
  cali_slopes = NULL
  cali_intercepts = NULL
  auc_rocs = NULL
  cv_res = NULL

  final_fitted_m = PRVT__train_federated_logreg_ensemble(fed_dfl, training_params = training_params)
  training_params$final_fitted_m = final_fitted_m # this will be the return obj passed through

  if(!with_bootstrap)
    bootstrap_n_inter = 1

  for(i in 1:bootstrap_n_inter) {
    try_log_info("@try_bootstrap_kfold_fl_ensemble - ::::::::::::::::: BOOTSTRAP iteration %d", i)
    c_res = try_kfold_cv_fl_ensemble(fed_dfl, training_params = training_params)
    if(is.null(preds)) {
      preds = c_res$cv_results$cv_test_preds
      true_vals = c_res$cv_results$cv_test_true_vals
    }
    else
      preds = preds + c_res$cv_results$cv_test_preds

    cali_slopes = c(cali_slopes, c_res$cv_results$cv_perf_metrics$cali_slope)
    cali_intercepts = c(cali_intercepts, c_res$cv_results$cv_perf_metrics$cali_intercept)
    auc_rocs = c(auc_rocs, c_res$cv_results$cv_perf_metrics$aucroc)
    cv_res = c_res
  }
  preds = preds / bootstrap_n_inter
  if(with_bootstrap)
    final_fitted_m$bootstrap_results = list(preds = preds, true_vals = true_vals,
                                            cali_slopes = cali_slopes, cali_intercepts = cali_intercepts, auc_rocs = auc_rocs)
  if(with_cv)
    final_fitted_m$cv_results = cv_res$cv_results

  return(final_fitted_m)
}

try_kfold_cv_fl_ensemble <- function(fed_dfl, training_params) {
  # init params
  list2env(training_params, envir = environment())
  final_fitted_m = training_params$final_fitted_m

  train_fed_dfl = NULL
  test_fed_dfl = NULL

  if(with_cv) {
    for(c_nm in names(fed_dfl)) {
      c_df = fed_dfl[[c_nm]]
      c_folds =  create_kfolds(c_df, cv_n_folds)
      for(i in 1:cv_n_folds) {
        if(len(train_fed_dfl) < i)  {
          train_fed_dfl[[i]] = list()
          test_fed_dfl[[i]] = list()
        }
        train_fed_dfl[[i]][[c_nm]] = c_folds$train_dfs[[i]]
        test_fed_dfl[[i]][[c_nm]] = c_folds$test_dfs[[i]]
      }
    } # this is now [[fold]][[center]]
  }

  c_preds = NULL
  c_true_vals = NULL
  if(!with_cv) {
    cv_n_folds = 1
    train_fed_dfl = list(fed_dfl)
    test_fed_dfl = list(fed_dfl)
  }


  for(c_fold in 1:cv_n_folds) {
    c_ens = PRVT__train_federated_logreg_ensemble(train_fed_dfl[[c_fold]], training_params = training_params)
    c_all_test_rows = do.call("rbind", test_fed_dfl[[c_fold]])
    c_preds = c(c_preds, c_ens$predict(newdata = c_all_test_rows))
    c_true_vals = c(c_true_vals, c_all_test_rows$outcome_of_interest)
  }
  c_perf_metrics = NULL
  if(cv_with_perf_metrics)
    c_perf_metrics = compute_perf_metrics(c_preds, c_true_vals)

  cv_res = list()
  cv_res$cv_perf_metrics = c_perf_metrics
  cv_res$cv_test_preds = c_preds
  cv_res$cv_test_true_vals = c_true_vals
  final_fitted_m$cv_results = cv_res


  return(final_fitted_m)
}

PRVT__train_federated_logreg_ensemble <- function(fed_dfl, training_params) {
  # init params
  list2env(training_params, envir = environment())

  l_pms = list()
  vars_selected_df = NULL
  if("data.frame" %in% class(vars_selected))
    vars_selected_df = vars_selected
  # iterate over each partition dataset (which has variables selected)
  partition_names = if(is.null(vars_selected_df)) names(fed_dfl) else sort(uniq(vars_selected_df$center))
  for(c_nm in names(fed_dfl)) {
    if(len(vars_selected_df) != 0) vars_selected = vars_selected_df[vars_selected_df$center == c_nm,]$vars
    c_df = fed_dfl[[c_nm]]
    try_log_debug("***************************** START partition %s *************************** ", c_nm)
    lasso_model = NULL
    if(do_feature_selection_per_partition) {
      if(len(vars_selected) != 0)
        try_log_warn("@PRVT__train_federated_logreg_ensemble - do_feature_selection_per_partition is true, but vars_selected is also provided.
                   -- going to ignore contents of vars_selected")
      c_df = remove_columns_with_non_full_conf_matrix(c_df)
      x = model.matrix(outcome_of_interest~., c_df)[,-1]
      y = c_df$outcome_of_interest
      if(any(table(y) <= 8)) {
        try_log_warn("SKIPPING model building for parition %s. Reason:  number of yes or no cases too little (<8)", c_nm)
        next
      }

      cv.lasso <- glmnet::cv.glmnet(x, y, alpha = 1, family = "binomial")
      # plot(cv.lasso)
      # cv.lasso$lambda.min
      glmnetfit1 = glmnet::glmnet(x, y, family = "binomial", alpha = 1, lambda = cv.lasso$lambda.min, maxit =1000000  )
      lasso_model = glmnetfit
      # class(coef(glmnetfit1))
      # print(coef(glmnetfit1)[,1])
      # note: these two model fittings produce slightly differnt coefficients! (non-lasso one seemed slightly higher auc)
      vars_selected = names(which(coef(glmnetfit1)[,1] != 0))[-1]
      try_log_debug("LASSO vars selected = %s", paste0(vars_selected,  collapse = ", "))
      lasso_preds = as.vector( logit2prob( predict(glmnetfit1, x) )  )
      try_log_debug("glmnetfit AUC = %0.3f", try_get_aucroc_ci(lasso_preds, y)[2] )
    }

    print(table(c_df$outcome_of_interest))
    c_df = autofill_missing_var_columns(c_df, vars_selected)
    if(do_feature_selection_per_partition)
      l_pms[[c_nm]] = lasso_model
    else
      l_pms[[c_nm]] = train_and_pool_logreg(list(c_df), vars_selected, with_pooling = F)[[1]]
    # lr_preds = logit2prob(  predict(c_lr_fit, c_df) )
    # try_log_debug("LR fit [center%s] train AUC = %0.3f",c_nm, try_get_aucroc_ci(lr_preds, c_df$outcome_of_interest)[2] )
  }
  ensemble_s3 = list(models = l_pms)
  class(ensemble_s3) = c(FEDERATION_MECHANISM_ENSEMBLE_OF_MODELS, "lr", class(ensemble_s3 ))

  predict_ensemble <- function(ens_s3, newdata, weigthing_scheme) {
    m_preds_df = NULL
    for(c_nm in names(ens_s3$models)) {
      m_fit = ens_s3$models[[c_nm]]
      # heuristic to detect models that were not well fitted - skip them from computation
      if(UOF_VN %in% names(m_fit$coefficients)) {
        try_log_debug("skipping center %s (model not well fitted)", c_nm)
      }
      else {
        m_preds = logit2prob(predict(m_fit, newdata = newdata))
        y = if(len(newdata$outcome_of_interest) == 0) rep(NA,len(m_preds)) else newdata$outcome_of_interest
        m_preds_df = rbind(m_preds_df, as.data.frame(list(prediction = m_preds, true_val = y, ens_idx = c_nm,
                                                          record_idx = 1:len(m_preds))))
      }
    }

    fused_preds = NULL
    if(weigthing_scheme == "mean")
      fused_preds = umap(1:nrow(newdata), function(x) { mean(m_preds_df[m_preds_df$record_idx == x, ]$prediction) })
    return(fused_preds)
  }
  ensemble_s3$weighting_scheme = weighting_scheme

  ensemble_s3$predict <- function(newdata) {
    return(predict_ensemble(ensemble_s3, newdata, ensemble_s3$weighting_scheme))
  }
  return(ensemble_s3)
}

try_LASSO_local_models <- function(x_train, y_train) {
  x_train = as.data.frame(x_train)
  x_train$outcome_of_interest = as.numeric(y_train)-1
  # lasso_models_list = list()
  lasso_vars_list = list()
  for(cc in sort(uniq(x_train[, UOF_VN]))) {
    x_train_partition = x_train[x_train[, UOF_VN] == cc,]
    x_train_partition[, UOF_VN] = NULL
    y_train_partition = x_train_partition$outcome_of_interest
    y_train_partition = as.factor(y_train_partition)

    x_train_partition = remove_columns_with_non_full_conf_matrix(x_train_partition, remove_almostempty_quadrants = T ,
                                                                 cutoff_thresh = MIN_N_POS_OUTCOMES_IN_DATASET)
    x_train_partition$outcome_of_interest = NULL

    try_log_debug("center %s running LASSO feature selection...", cc)

    c_lasso_vars = try_LASSO(data.matrix(x_train_partition), y_train_partition, return_model = F, treat_each_category_as_seperate_var = F)
    lasso_vars_list[[cc]] = c_lasso_vars
  }


  return(lasso_vars_list)
}

try_training_local_models_glm <- function(df_train, vars_selected, df_test) {
  glm_models_list = list()
  if(is.factor(df_train[, UOF_VN]))
    df_train[, UOF_VN] = as.numeric(levels(df_train[, UOF_VN]))[df_train[, UOF_VN]]
  for(cc in sort(uniq(df_train[, UOF_VN]))) {
    cat("****@try_training_local_models_glm CENTER   ")
    cat(cc)
    cat("\n")
    if(cc > len(vars_selected))
      next
    c_vars_selected = vars_selected[[cc]]
    if('(Intercept)' %in% c_vars_selected)
      c_vars_selected = c_vars_selected[-which(c_vars_selected == '(Intercept)')]
    if(len(c_vars_selected) == 0)
      next
    c_train = df_train[df_train[, UOF_VN] == cc,]
    c_test = df_test[df_test[, UOF_VN] == cc,]

    tmp = prep_x_matricies(c_train, c_test)
    x_train = tmp$x_train
    y_train = c_train$outcome_of_interest

    x_train[,grep(UOF_VN, cns(x_train), val = T)] = NULL
    x_train[, UOF_VN] = c_train[, UOF_VN]

    x_train = as.data.frame(x_train)
    cols_to_remove = cns(x_train)[-which(cns(x_train) %in% c_vars_selected)]
    x_train[,cols_to_remove] = NULL

    x_train$outcome_of_interest = c_train$outcome_of_interest
    x_train = data.matrix(remove_columns_with_non_full_conf_matrix(x_train, remove_almostempty_quadrants = T ,
                                                                   cutoff_thresh = MIN_N_POS_OUTCOMES_IN_DATASET))
    x_train = as.data.frame(x_train)
    x_train$outcome_of_interest = y_train
    c_vars_selected = c_vars_selected[which(c_vars_selected %in% cns(x_train))]

    x_test = tmp$x_test
    y_test = c_test$outcome_of_interest

    x_test = as.data.frame(x_test)
    cols_to_remove = cns(x_test)[-which(cns(x_test) %in% c_vars_selected)]
    x_test[,cols_to_remove] = NULL
    x_test$outcome_of_interest = c_test$outcome_of_interest

    c_glm = train_and_pool_logreg(list(x_train),  c_vars_selected, with_pooling = F, verbose = T)
    if(len(c_glm) == 1) {
      train_preds = predict(c_glm[[1]], newdata = x_train, type = 'response')
      y_train_num = as.numeric(levels(y_train)[y_train])
      tryCatch({
        train_cali = try_get_calibration(train_preds, y_train_num, verbose = F)

        if(is.data.frame(x_test) && nrow(x_test) > 10) {
          test_preds = predict(c_glm[[1]], newdata = x_test, type = 'response')
          y_test_num = as.numeric(levels(y_test)[y_test])

          test_cali = try_get_calibration(test_preds, y_test_num, verbose = F)
          try_log_debug("^0000000^  CENTER: %s, train/test cali int := %0.3f/%0.3f", cc, train_cali$cal.intercept, test_cali$cal.intercept)
        }
        else {
          try_log_debug("^0000000^  CENTER: %s,CANNOT COMPUTE TEST  PREDS", cc)
        }
      }, error = function(err) {
        # Code block to handle errors
        # and specific exceptions
      }, warning = function(wrn) {
        # Code block to handle warnings
      }, finally = {
        # Code block to be executed
        # regardless of errors or warnings
      })
    }

    if(len(c_glm) != 0) {
      class(c_glm) = c(class(c_glm), sprintf("center_%d", as.numeric(cc)))
      glm_models_list[[len(glm_models_list) + 1]] = c_glm
    }
  }

  #   c_coefs = coef(c_lasso_model)[,c_lasso_model$lambda2use_idx]
  #   c_vars_selected = names(c_coefs)[which(as.numeric(c_coefs) != 0)]
  #   n_vars = len(c_vars_selected) - 1
  #   if(n_vars < 2){
  #     try_log_warn("Could not derive local model for center  %d", cc)
  #   }
  #   else {
  #     try_log_debug("center %s, selected  %d vars", cc, len(c_vars_selected)-1)
  #     class(c_lasso_model) = c(class(c_lasso_model), sprintf("center_%d", cc))
  #     lasso_models_list[[len(lasso_models_list) + 1]] = c_lasso_model
  #   }
  # }
  class(glm_models_list) = c(class(glm_models_list), FEDERATION_MECHANISM_LOCAL_MODELS_ONLY)
  return(glm_models_list)
}

try_fit_ensemble_and_lr_of_predictions <- function(df_train, vars_selected, df_test) {
  glm_models_list = try_training_local_models_glm(df_train, vars_selected, df_test)

  x_train = prep_x_matrix(df_train, do_remove_columns_with_non_full_conf_matrix = F, remove_almostempty_quadrants = F)
  y_train = df_train$outcome_of_interest
  x_train =  as.data.frame(x_train)
  x_train[,grep(UOF_VN, cns(x_train), val = T)] = NULL
  x_train[, UOF_VN] = df_train[, UOF_VN]


  # note: you are sharing the outcomes between partitions!
  preds_df = try_get_preds_df_ensemble(glm_models_list, x_train, y_train, is_glm = T)

  preds_df[, UOF_VN] = x_train[, UOF_VN]
  preds_df[, UOF_VN] = as.numeric(levels(preds_df[, UOF_VN])[preds_df[, UOF_VN]])
  lr_fit = NULL


  center_weights = NULL
  # in a way you are doing a centralized L2 recalibration like this
  if(STACKED_PERFORM_L2_RECALL)
    lr_fit =  train_and_pool_logreg(list(preds_df), cns(preds_df)[-len(cns(preds_df))], with_pooling = F, verbose = T)[[1]]
  if(!STACKED_PERFORM_L2_RECALL) {
    center_volumes = list()
    for(c_center in uniq(preds_df[, UOF_VN])){
      c_center_clm = sprintf("center_%d", c_center)
      center_volumes[[c_center_clm]] = howmany(preds_df[, UOF_VN] == c_center)
    }
    center_volumes_to_keep = c()
    for(cent_coef_nm in cns(preds_df)[grep('center_', cns(preds_df))]) {
      if(len(grep('center_',cent_coef_nm)) == 0)
        next
      center_volumes_to_keep = c(center_volumes_to_keep, cent_coef_nm)
    }
    tot_vol = 0
    for(c_cente_nm in names(center_volumes)) {
      if(!c_cente_nm %in% center_volumes_to_keep)
        center_volumes[[c_cente_nm]] = NULL
      else
        tot_vol = tot_vol + center_volumes[[c_cente_nm]]
    }

    center_weights = list()
    for(c_cente_nm in names(center_volumes)) {
      center_weights[[c_cente_nm]] = center_volumes[[c_cente_nm]] / tot_vol
    }
  }
  if(STACKED_PERFORM_L1_RECALL) {
    x34 = 456
    # lr_fit =  train_and_pool_logreg(list(preds_df), cns(preds_df)[-len(cns(preds_df))], with_pooling = F, verbose = T)[[1]]
    # 1. get non-calibrated preds
    cent_cols = cns(preds_df)[grep("center_",cns(preds_df))]
    non_cal_preds = umap(1:nrow(preds_df), function(x){
      sum(umap(cent_cols, function(y) { preds_df[x,y] * center_weights[[y]] }))
    })
    # 2. run l1 recal on them
    l1recal_res = perform_L1_recalibration(non_cal_preds, outs = preds_df$outcome_of_interest)
    intercept = l1recal_res$recal_model$coefficients[1]
    slope = l1recal_res$recal_model$coefficients[2]

    pooled_recal_predict_fn <- function(new_preds) {
      # Calibrate the probabilities using the logistic regression equation
      calibrated_preds <- plogis(intercept + slope * new_preds)
      return(calibrated_preds)
    }
  }


  # lr_fit_preds = predict(lr_fit, newdata =  preds_df)
  # try_get_aucroc_ci(lr_fit_preds, y_train)[2]

  ret_recal_fn = if(STACKED_PERFORM_L1_RECALL) pooled_recal_predict_fn else NULL
  return_obj = list(lasso_models_list = glm_models_list,
                    lr_from_local_models = lr_fit, center_weights = center_weights, recal_fn = ret_recal_fn)
  class(return_obj) = c(class(return_obj), FEDERATION_MECHANISM_ENSEMBLE_OF_MODELS_WITH_LR_FIT_ON_PREDICTIONS)
  return(return_obj)
}

try_training_global_model_glm <- function(df_train, vars_selected, df_test) {
  c_vars_selected = vars_selected
  if('(Intercept)' %in% c_vars_selected)
    c_vars_selected = c_vars_selected[-which(c_vars_selected == '(Intercept)')]
  c_train = df_train
  c_test = df_test
  tmp = prep_x_matricies(c_train, c_test)
  x_train = tmp$x_train
  y_train = c_train$outcome_of_interest
  x_train[,grep(UOF_VN, cns(x_train), val = T)] = NULL
  x_train[, UOF_VN] = c_train[, UOF_VN]

  x_train = as.data.frame(x_train)
  cols_to_remove = cns(x_train)[-which(cns(x_train) %in% c_vars_selected)]
  x_train[,cols_to_remove] = NULL

  x_train$outcome_of_interest = c_train$outcome_of_interest
  x_train = data.matrix(remove_columns_with_non_full_conf_matrix(x_train, remove_almostempty_quadrants = T ,
                                                                 cutoff_thresh = MIN_N_POS_OUTCOMES_IN_DATASET))
  x_train = as.data.frame(x_train)
  x_train$outcome_of_interest = y_train
  c_vars_selected = c_vars_selected[which(c_vars_selected %in% cns(x_train))]

  x_test = tmp$x_test
  y_test = c_test$outcome_of_interest

  x_test = as.data.frame(x_test)
  cols_to_remove = cns(x_test)[-which(cns(x_test) %in% c_vars_selected)]
  x_test[,cols_to_remove] = NULL
  x_test$outcome_of_interest = c_test$outcome_of_interest

  c_glm = train_and_pool_logreg(list(x_train), c_vars_selected, with_pooling = F, verbose = F)[[1]]

  class(c_glm) = c(class(c_glm), "global-lasso")
  return(c_glm)
}

train_lr_federated_with_test_df <- function(df_train, vars_selected, test_df = NULL) {
  if(len(test_df) == 0) {
    try_log_warn("@train_lr_federated_with_test_df - test_df is NULL, going to use df_train for computing metrics ")
    test_df = df_train
  }
  print("debug")

  return(train_lr_federated(df_train, vars_selected, test_df))
}


################################################  Synthetic datagen functions ################################################
# for(cvar in cns(df1)) {
#   cat(sprintf("%s : %s, %0.3f, %0.3f\n", cvar, class(df1[,cvar]), min(df1[,cvar], na.rm = T), max(df1[,cvar], na.rm = T)  ))
# }
try_sample_continu <- function(n, minv = 0, maxv = 1, missing_prop = 0.02) {
  samples = runif(n, minv, maxv)
  miss_idxs = sample(1:n, round(n*missing_prop), replace = F)
  samples[miss_idxs] = NA
  return(samples)
}

try_sample_int <- function(n, minv = 0, maxv = 1, missing_prop = 0.02) {
  return(round(try_sample_continu(n, minv, maxv, missing_prop)))
}

try_sample_cat <- function(n, cats, missing_prop = 0.02) {
  samples = sample(cats, n, replace = T)
  miss_idxs = sample(1:n, round(n*missing_prop), replace = F)
  samples[miss_idxs] = NA
  return(samples)
}

try_gen_synthetic_TAVI_data <- function(n = 10000) {

  synth_data = as.data.frame( list (
    leeftijd = try_sample_int(n, 18, 100),
    geslacht = try_sample_int(n),
    kreatinine_gehalte = try_sample_int(n, 1, 1000),
    diabetes = try_sample_cat(n, c(0, 1, 2, 10, 20, 30, 90)),
    LVEF = try_sample_int(n, 6, 90),
    PA_druk = try_sample_int(n, 10, 120),
    chr_longziekte = try_sample_int(n),
    art_vaatpathologie = try_sample_int(n),
    neuro_disfunctie = try_sample_int(n),
    cardiochir_eerder = try_sample_int(n),
    endocarditis = try_sample_int(n),
    krit_preop_toestand = try_sample_int(n),
    instabiele_AP = try_sample_int(n),
    recent_MI = try_sample_int(n),
    thorac_aortachir = try_sample_int(n),
    postinfarct_VSR = try_sample_int(n),
    dialyse = try_sample_int(n),
    slechte_mobiliteit = try_sample_int(n),
    NYHA = try_sample_int(n, 1, 4),
    CCS_IV = try_sample_int(n),
    urgentie = try_sample_cat(n, c(10, 20, 30, 40)),
    interv_gewicht = try_sample_cat(n, c(10, 11, 20, 30)),
    CVA_eerder = try_sample_int(n),
    aklepchir_eerder = try_sample_int(n),
    PM_eerder = try_sample_int(n),
    narcose = try_sample_int(n),
    TAVI_toegang = try_sample_cat(n, c(10, 11, 12, 15, 25, 30, 90)),
    TAVI_predilatatie = try_sample_int(n),
    TAVI_postdilatatie = try_sample_int(n),
    CVA_restletsel = try_sample_int(n),
    re_aklep = try_sample_int(n),
    eGFR = try_sample_continu(n, 2, 300),
    centrum = try_sample_int(n, 1, 10, missing_prop = 0),
    jaar = try_sample_int(n, 2013, 2023),
    bmi = try_sample_continu(n, 10, 100),
    bsa = try_sample_continu(n, 1, 5),
    mort_status_30d = try_sample_int(n, missing_prop = 0)
  ))
  # insert some slight correlation between outcome and 2 vars
  synth_data_cases = synth_data[synth_data$mort_status_30d == 1,][1:round(n/4),]
  nas_idx1 = which(is.na(synth_data_cases$LVEF))
  synth_data_cases$LVEF[nas_idx1] = min(synth_data_cases$LVEF, na.rm = T)
  nas_idx2 = which(is.na(synth_data_cases$eGFR))
  synth_data_cases$eGFR[nas_idx2] = min(synth_data_cases$eGFR, na.rm = T)

  synth_data_cases$LVEF = max(6, synth_data_cases$LVEF - sample(1:100/1000, nrow(synth_data_cases), replace = T))
  synth_data_cases$eGFR = max(2, synth_data_cases$eGFR - sample(1:100/1000, nrow(synth_data_cases), replace = T))
  synth_data_cases$LVEF[nas_idx1] = NA
  synth_data_cases$eGFR[nas_idx2] = NA
  synth_data[synth_data$mort_status_30d == 1,][1:round(n/4),] = synth_data_cases

  # View(try_round(cor(synth_data)[37,], 2))
  return(synth_data)

}
