# Author: T.Y.
# Date created: 15.09.2022


################################################  LASSO functions ################################################

try_LASSO_ensemble <- function(x_train, y_train) {
  x_train = as.data.frame(x_train)
  x_train$outcome_of_interest = as.numeric(y_train)-1
  lasso_models_list = list()

  for(cc in sort(uniq(x_train[,UOF_VN]))) {
    x_train_partition = x_train[x_train[,UOF_VN] == cc,]
    x_train_partition[,UOF_VN] = NULL
    y_train_partition = x_train_partition$outcome_of_interest
    y_train_partition = as.factor(y_train_partition)

    x_train_partition = remove_columns_with_non_full_conf_matrix(x_train_partition, remove_almostempty_quadrants = T ,
                                                                 cutoff_thresh = MIN_N_POS_OUTCOMES_IN_DATASET)

    try_log_debug("center %s running LASSO feature selection...", cc)

    c_lasso_model = try_LASSO(data.matrix(x_train_partition), y_train_partition, return_model = T)


    c_coefs = coef(c_lasso_model)[,c_lasso_model$lambda2use_idx]
    c_vars_selected = names(c_coefs)[which(as.numeric(c_coefs) != 0)]
    n_vars = len(c_vars_selected) - 1
    try_log_debug("center %s, selected  %d vars", cc, len(c_vars_selected)-1)
    if(n_vars >= MIN_N_VARS_SELECTED_PER_MODEL_ENSEMBLE) {
      class(c_lasso_model) = c(class(c_lasso_model), sprintf("center_%d", cc))
      lasso_models_list[[len(lasso_models_list) + 1]] = c_lasso_model
    }

  }
  # IMPROVEMENT IDEA: once you create the ensemble, train a new model whose imputs are the predictions of each model from the ensemble
  class(lasso_models_list) = c(class(lasso_models_list), FEDERATION_MECHANISM_ENSEMBLE_OF_MODELS)
  return(lasso_models_list)
}

#@param agreement_strenth - proportion of partitions that should have selected a feature before it gets used in the aggregated list of features
try_LASSO_federated <- function(x_train, y_train, agreement_strength=TRAINING_PARAMS_FL$agrreement_strength_lasso, verbose = T) {
  x_train = as.data.frame(x_train)
  x_train$outcome_of_interest = as.numeric(y_train)-1

  vars_selected_df = NULL
  old_HYPERPARAM_LAMBDA_FN = HYPERPARAM_LAMBDA_FN
  for(cc in sort(uniq(x_train[,UOF_VN]))) {
    HYPERPARAM_LAMBDA_FN <<- sprintf("%s_%s", old_HYPERPARAM_LAMBDA_FN, cc)
    x_train_partition = x_train[x_train[,UOF_VN] == cc,]
    x_train_partition[,UOF_VN] = NULL
    y_train_partition = x_train_partition$outcome_of_interest
    y_train_partition = as.factor(y_train_partition)

    x_train_partition = remove_columns_with_non_full_conf_matrix(x_train_partition, remove_almostempty_quadrants = T ,
                                                                   cutoff_thresh = MIN_N_POS_OUTCOMES_IN_DATASET)
    if(verbose)
      try_log_debug("@try_LASSO_federated - center %s (aggr strenth = %0.2f)", cc, agreement_strength)
    c_vars_selected = try_LASSO(data.matrix(x_train_partition), y_train_partition,
                                return_model = F, treat_each_category_as_seperate_var = F, verbose = verbose)
    if(len(c_vars_selected) > 0)
      vars_selected_df = rbind(vars_selected_df,
                             as.data.frame(list(vars = c_vars_selected, center = rep(cc, len(c_vars_selected)))))

  }
  HYPERPARAM_LAMBDA_FN <<- old_HYPERPARAM_LAMBDA_FN
  # 06.Apr.2023 - weighted average of variables (based on #TAVIs per center)
  centers_with_selected_vars = sort(uniq(vars_selected_df$center))
  center_voting_weights = ( nuniq(vars_selected_df$center)*table(x_train[x_train[,UOF_VN] %in% centers_with_selected_vars,][,UOF_VN]) ) / nrow(x_train)
  # sum(center_voting_weights) == nuniq(vars_selected_df$center)
  vars_selected_df$voting_weight = 0
  for(c_cent in names(center_voting_weights)){
    c_weight = center_voting_weights[[c_cent]]
    vars_selected_df[vars_selected_df$center == c_cent,]$voting_weight = c_weight
  }

  n_partitions = nuniq(vars_selected_df$center)
  vars_counts = table(vars_selected_df$vars) # old way, non weighted
  # new way - weighted
  for(c_var in uniq(vars_selected_df$vars)) {
    c_weighted_count = sum(vars_selected_df[vars_selected_df$vars == c_var,]$voting_weight)
    vars_counts[[c_var]] = c_weighted_count
  }
  min_n_counts_needed = n_partitions * agreement_strength

  aggregated_vars_selected = names(which(vars_counts >= min_n_counts_needed))
  if(verbose && len(aggregated_vars_selected) < 3)
    try_log_warn("@try_LASSO_federated -  len(aggregated_vars_selected) = %d !!!", len(aggregated_vars_selected))
  return(aggregated_vars_selected)
}

try_LASSO_dummy_allvars_returned <- function(x_train, y_train) {
  return(setdiff( cns(x_train),  grep_on_colnames(x_train, UOF_VN) ))
}

try_LASSO <- function(x_train, y_train, return_model = T, maxit = 10000, treat_each_category_as_seperate_var = F, verbose = T) {
  n_outcomes_pos = howmany(as.numeric(y_train) == max(as.numeric(y_train)) )
  n_outcomes_pos = if(len(uniq(y_train )) == 1) 0 else n_outcomes_pos
  if(n_outcomes_pos < MIN_N_POS_OUTCOMES_IN_DATASET) {
    if(verbose)
      try_log_debug("@try_LASSO - skip run as not enough outcome occurrences (%d/%d)", len(y_train), n_outcomes_pos)
    return(NULL)
  }
  if("outcome_of_interest" %in% cns(x_train))
    x_train = x_train[, - which(cns(x_train) == "outcome_of_interest")]
  if(UOF_VN %in% cns(x_train))
    x_train = x_train[, - which(cns(x_train) == UOF_VN)]

  cv_stratified_folds = splitTools::create_folds(y_train, k = 10, type = "stratified")
  foldids = rep(-1, nrow(x_train))
  for(c_fold in 1:len(cv_stratified_folds)){
    c_entries = which(!(1:nrow(x_train) %in% cv_stratified_folds[[c_fold]]))
    foldids[c_entries] = c_fold
  }
  # -> LASSO control will have 10-fold CV of df_train for selecting its hyperparam
  cv.lasso <- glmnet::cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", nfolds = 10, foldid = foldids) # alpha 1 = lasso / 0 = ridge

  lambdas_to_try = cv.lasso$lambda
  best_lamb_idx = last(which(cv.lasso$lambda == cv.lasso$lambda.1se))

  glmnetfit1 = glmnet::glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = lambdas_to_try, maxit = maxit)
  # there will be 3 lambdas here : lambda.se1, lambda_mean[min/se1], lambda.min - 1st will give you less varibales, last might give you all varibles,

  MIN_VARS_NEEDED = 2
  vars_used = NULL
  if(!all(glmnetfit1$beta == 0)) {
    betas_df = as.data.frame(t(data.matrix(glmnetfit1$beta)))
    nvars_per_labmda = umap(1:nrow(betas_df), function(x){  howmany(abs(betas_df[x,]) > 0) })
    lambdas_with_sufficient_nonzero_betas_idxs = which(nvars_per_labmda >= MIN_VARS_NEEDED)
    glmnetfit1$lambda2use_idx = best_lamb_idx
    will_next_command_work = len(glmnetfit1) !=0 && 'lambda2use_idx' %in% names(glmnetfit1) && ncol(coef(glmnetfit1)) >= glmnetfit1$lambda2use_idx
    if(will_next_command_work) {
      coefs = coef(glmnetfit1)[,glmnetfit1$lambda2use_idx]
      will_next_command_work = len(coefs) != 0 && any(coefs != 0)
      if(will_next_command_work) {
        vars_used = coefs[which(coefs != 0)]
      }
      else {
        print("Failed to find any non-zero betas in LASSO!")
        return(NULL)
        }
    }
    else {
        print("Failed to find any non-zero betas in LASSO!")
        return(NULL)
      }

  }
  else {
    print("no non-zero betas left!")
  }

  vars_used = names(vars_used)
  if(verbose)
    try_log_debug("@try_LASSO - selected %d vars: %s", len(vars_used), paste0(c("\n -> ",vars_used), collapse = "\n -> "))
  if(return_model)
    return(glmnetfit1)
  if(!treat_each_category_as_seperate_var)
  {
    all_coef_names = names(coef(glmnetfit1)[,glmnetfit1$lambda2use_idx])
    for(var_used in vars_used){
      for(i in POSSIBLE_VAR_CATEGORIES){
        if(len(i) > 1 && var_used %in% i)
          vars_used = union(vars_used, i)
    }}
  }
  return(vars_used)
}

try_LASSO_and_model_training <- function(x_train, y_train) {
  return(try_LASSO(x_train, y_train, return_model = T))
}

try_LASSO_variable_selection_only <- function(x_train, y_train) {
  return(try_LASSO(x_train, y_train, return_model = F, treat_each_category_as_seperate_var = F))
}

try_LASSO_model_training_wrapper <- function(df_train, trained_model, test_df = NULL) {
  return(trained_model)
}


################################################  AIC functions (unused) ################################################

try_backwards_AIC <- function(x_train, y_train) {
  # iii) using df_train select variables via LASSO
  # -> LASSO control will have 10-fold CV of df_train for selecting its hyperparam
  xx = as.data.frame(x_train)
  xx$outcome_of_interest = y_train
  vars_selected = perform_backwards_AIC(xx)
  return(vars_selected)
}

perform_backwards_AIC_federated_partial <- function(params_list) {
  return(perform_backwards_AIC_federated(fed_dfl, params_list = params_list))
}

#  @params_list - overrides other params
perform_backwards_AIC_federated <- function(fed_dfl, n_steps = 4, agreement_strength = 0.25, count_each_category_as_seperate_variable = T,
                                            params_list = NULL) {

  if(!is.null(params_list)){
    n_steps = params_list$n_steps
    agreement_strength = params_list$agreement_strength
    if(!is.null(params_list$count_each_category_as_seperate_variable))
      count_each_category_as_seperate_variable = params_list$count_each_category_as_seperate_variable
  }


  any_vars_left_to_remove = T
  n_centers = len(fed_dfl)
  vars_removed = NULL
  n_vars_left_df = as.data.frame(list(center = names(fed_dfl), no_more_aic = rep(F, len(fed_dfl))))
  while(any_vars_left_to_remove) {

    vars_removed_per_center = NULL
    print(table(n_vars_left_df$no_more_aic))
    print(vars_removed)
    if(all(n_vars_left_df$no_more_aic) ||  len(vars_removed) == len(cns(fed_dfl[[1]]))-1  )
      break

    for(c_nm in names(fed_dfl)){

      if(n_vars_left_df[n_vars_left_df$center  == c_nm, ]$no_more_aic)
        next

      c_df = fed_dfl[[c_nm]]

      if(!is.null(vars_removed))
        c_df[, vars_removed] = NULL

      c_vars = setdiff(cns(c_df), "outcome_of_interest")
      n_vars = perform_backwards_AIC(c_df, count_each_category_as_seperate_variable = count_each_category_as_seperate_variable, n_steps = n_steps)
      c_vars_removed = setdiff(c_vars, n_vars)
      if(len(c_vars_removed) > 0)
        vars_removed_per_center =  rbind(vars_removed_per_center,  as.data.frame(list(center = rep(c_nm, len(c_vars_removed)), removed_vars = c_vars_removed)))
      else
        n_vars_left_df[n_vars_left_df$center  == c_nm, ]$no_more_aic = T
    }
    vars_to_remove_table = table(vars_removed_per_center$removed_vars)
    new_vars_removed = names(which(vars_to_remove_table >= n_centers*agreement_strength))
    if(len(new_vars_removed) < 1)  {
      try_log_warn("@perform_backwards_AIC_federated - could not find any more vars to remove with sufficient agreemnt strength %0.2f [max found %0.2f] - stopped before convergance",
                   agreement_strength, max(vars_to_remove_table)/n_centers)
      break
    }
    vars_removed = c(vars_removed, new_vars_removed)
    if(len(vars_removed) != uniq(len(vars_removed))) {
      print("woooooow")
      print(table(vars_removed))
      print("woooooow")
      print("woooooow")
    }
  }
  vars_selected  = setdiff(cns(fed_dfl[[1]]), c(vars_removed, "outcome_of_interest"))
  try_log_debug("selected %d vars from %d candidates", len(vars_selected), len(cns(fed_dfl[[1]])) )
  return(setdiff(cns(fed_dfl[[1]]),c(vars_removed, "outcome_of_interest" )  ))
}

