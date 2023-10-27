# Author: T.Y.
# Date created: 15.09.2022


# at this point hyperparam values are not yet selected, first we do that then we fit the model
PRVT__train_federated_logreg_model_update <- function(fed_dfl, training_params, use_ks_test = F) {
  if(RUN_HYPERPARAM_OPTIMIZATION) {
    print(sprintf("OPTIMIZE HYPERPARAMS using %s", HYPERPARAM_METRIC))
    params_grid = expand.grid(list(learning_rate = training_params$learning_rate,
                                   epochs = training_params$epochs,
                                   agrreement_strength_lasso = training_params$agrreement_strength_lasso ))
    eval_metric_fn_mapper = list(mse = function(x,y) mean((x-y)^2),
                                 inv.auc = function(x,y) 1-try_get_aucroc(predictions = x, true_vals = y) )
    eval_metric_fn = eval_metric_fn_mapper[[HYPERPARAM_METRIC]]
    # expand grid rows x n_cv_folds
    n_folds_hyperparam_cv = 5
    exp_params_grid = NULL # like params_grid but multiplied rows times num of k folds
    for(c_f in 1:n_folds_hyperparam_cv) {
      c_params_grid = params_grid
      c_params_grid$hyper_fold = c_f
      exp_params_grid = rbind(exp_params_grid, c_params_grid)
    }
    params_grid[,HYPERPARAM_METRIC] = 99999
    exp_params_grid[,HYPERPARAM_METRIC] = 99999
    exp_params_grid$tie_settler_mse = 999999
    params_grid$tie_settler_mse =  999999
    test_df = training_params$test_df
    verbose = training_params$verbose

    # each df in fed dfl should not be mixed with the other
    # for each df in fed_dfl
    #     split c_df into 5 folds 80/20 train/validate
    # output -> 5 fed_dfls to train on + 5 fed_dfls to validate on
    train_fed_dfls = list()
    validate_fed_dfls = list()

    fed_dfl_clean = list()
    # in lcoa, exactly one partition will be missing each time from the training set
    # and one only will be present in test set
    for(i in 1:len(fed_dfl)){
      c_df = fed_dfl[[i]]
      if(len(c_df) != 0)
        fed_dfl_clean[[len(fed_dfl_clean) + 1]] = c_df
    }
    fed_dfl = fed_dfl_clean
    for(i in 1:len(fed_dfl)) {
      c_df = fed_dfl[[i]]
      c_folds = create_kfolds(c_df, n_folds = n_folds_hyperparam_cv, verbose = F)
      for(c_epoch in 1:n_folds_hyperparam_cv){
        # c_epoch = fold , i = center
        if(i == 1) {
          train_fed_dfls[[c_epoch]] = list()
          validate_fed_dfls[[c_epoch]] = list()
        }
        train_fed_dfls[[c_epoch]][[i]] = c_folds$train_dfs[[c_epoch]]
        validate_fed_dfls[[c_epoch]][[i]] = c_folds$test_dfs[[c_epoch]]
      }
    }
    #
    for(c_hyper_fold in 1:n_folds_hyperparam_cv) {
      try_log_debug("::c_hyper_fold = %d", c_hyper_fold)

      c_train_fed_dfl = train_fed_dfls[[c_hyper_fold]]
      c_val_fed_dfl = validate_fed_dfls[[c_hyper_fold]]
      df_val = NULL
      for(i in 1:len(c_val_fed_dfl)) {
        df_val = rbind(df_val, c_val_fed_dfl[[i]])
      }

      for(i in 1:nrow(params_grid)){
        try_log_debug("::hyperparam optimize: Running row %d out of %d of params_grid", i, nrow(params_grid))
        c_params = params_grid[i,]

        training_params$learning_rate = c_params$learning_rate
        training_params$epochs = c_params$epochs
        training_params$agrreement_strength_lasso = c_params$agrreement_strength_lasso
        training_params$verbose = F
        training_params$test_df = NULL

        c_fit = PRVT__train_federated_logred_model_update_hyperparams_chosen(c_train_fed_dfl, training_params)
        if(len(c_fit) == 0)
          next
        c_preds = try_predict(c_fit, df_val)
        c_HYPERPARAM_METRIC_val = eval_metric_fn(c_preds, df_val$outcome_of_interest)
        # HYPERPARAM_METRIC
        fold_offset = min(which(exp_params_grid$hyper_fold == c_hyper_fold)) - 1
        c_row_idx = i + fold_offset
        exp_params_grid[c_row_idx,HYPERPARAM_METRIC] = c_HYPERPARAM_METRIC_val
        exp_params_grid[c_row_idx, ]$tie_settler_mse = mean((c_preds - df_val$outcome_of_interest)^2)

      }

      print(c_hyper_fold)
    }
    # init done by here...

    if(use_ks_test) {
      final_grid = NULL
      for(c_params_idx in 1:nrow(params_grid)){
        c_params = params_grid[c_params_idx,]
        # offset + indx
        c_row_idxs = umap(1:n_folds_hyperparam_cv, function(c_f)  {min(which(exp_params_grid$hyper_fold == c_f)) - 1 + c_params_idx})
        c_metric_vals = exp_params_grid[c_row_idxs, HYPERPARAM_METRIC]
        c_metric_vals = umap(c_metric_vals, function(x) if(x == 99999) 1 else x) # todo: magic numbers are bad, consider replacing all with 1
        names(c_metric_vals) = sprintf("hf_%d", 1:n_folds_hyperparam_cv)
        c_fgrid = c_params
        c_fgrid = cbind(c_fgrid, t(c_metric_vals))
        c_fgrid$mean_metric = mean(c_metric_vals,na.rm = T)
        final_grid = rbind(final_grid, c_fgrid)
      }
      best_mean = min(final_grid$mean_metric)
      best_row = which(final_grid$mean_metric == best_mean)[1]
      best_metric_vals = final_grid[best_row,(1:n_folds_hyperparam_cv)+4]

      final_grid$kspval = 2
      for(c_r in 1:nrow(final_grid)){
        c_row = final_grid[c_r,]
        c_metric_vals = c_row[1,(1:n_folds_hyperparam_cv)+4]
        final_grid[c_r,]$kspval = ks.test(as.numeric(c_metric_vals), as.numeric(best_metric_vals))$p.value
      }
      best_row_equals = which(final_grid$kspval > 0.5) # very conservative pvalue, i.e. we are more sure they are the same
      try_log_info("Chose  %d hyperparam sets out of %d possible combinations", len(best_row_equals), nrow(params_grid) )
      for(c_row_idx in best_row_equals){
        best_params = final_grid[c_row_idx,]
        try_log_info(" [**%d**] Best hyperparams chosen with %s = %0.3f", c_row_idx,
                     HYPERPARAM_METRIC, best_params$mean_metric)
        try_log_info("%s %s",names(best_params), as.numeric(best_params))
      }
      best_fits = list()
      for(c_row_idx in best_row_equals){
        best_params = final_grid[c_row_idx,]
        training_params$learning_rate = best_params$learning_rate
        training_params$epochs =  best_params$epochs #best_params$epochs
        training_params$agrreement_strength_lasso = best_params$agrreement_strength_lasso
        training_params$verbose = verbose
        training_params$test_df = test_df
        c_best_fit = PRVT__train_federated_logred_model_update_hyperparams_chosen(fed_dfl, training_params)
        best_fits[[len(best_fits) + 1]] =  c_best_fit
      }
      class(best_fits) = c(class(best_fits), FEDERATION_MECHANISM_UPDATE_SHARING_LIST)
      return(best_fits)
    }
    else if(!use_ks_test) { # rank score / dispute solver = 1) take one with lower variance in metric, 2) one with lower MSE
      filler_ranks = rep(999999, n_folds_hyperparam_cv)
      names(filler_ranks) = sprintf("hf%s_rnk", 1:n_folds_hyperparam_cv)
      filler_ranks = as.data.frame(t(filler_ranks))
      filler_ranks$avg_rank = umap(1:nrow(filler_ranks), function(x) mean(as.numeric(filler_ranks[x,])))
      while(nrow(filler_ranks) < nrow(exp_params_grid)) {
          filler_ranks = rbind(filler_ranks, filler_ranks[1,])
      }
      exp_params_grid = cbind(exp_params_grid, filler_ranks)
      for(c_hp_fold in sort(uniq(exp_params_grid$hyper_fold))) {
        c_fold_grid = exp_params_grid[exp_params_grid$hyper_fold == c_hp_fold,]
        c_optim_vals = c_fold_grid[, HYPERPARAM_METRIC]
        ranks= rep(9999, len(c_optim_vals))
        sorted_uniq_optim_vals = sort(uniq(c_optim_vals), index.return = T) # $x $ix
        for(c_val in uniq(c_optim_vals)){
          c_val_idxs = which(c_optim_vals == c_val)
          c_val_uniq_idx = which(sorted_uniq_optim_vals$x == c_val)
          c_rank = sorted_uniq_optim_vals$ix[which(sorted_uniq_optim_vals$ix == c_val_uniq_idx)]
          ranks[c_val_idxs] = c_rank
        }
        c_ranks = ranks #sort(c_optim_vals, index.return = T)$ix problem with this is it will give differnt ranks for identical values
        hf_rnk_col_idxs = rev(ncol(c_fold_grid):(ncol(c_fold_grid)-n_folds_hyperparam_cv))
        hf_rnk_col_idxs = hf_rnk_col_idxs[- len(hf_rnk_col_idxs)] # remove avg_rnk
        c_fold_grid[, hf_rnk_col_idxs[c_hp_fold]] = c_ranks
        # exp_params_grid[exp_params_grid$hyper_fold == c_hp_fold,] = c_fold_grid
        params_grid[,cns(c_fold_grid)[hf_rnk_col_idxs[c_hp_fold]]] = 999999
        c_hp_m3k_nm = sprintf('%s-%d', HYPERPARAM_METRIC,c_hp_fold)
        params_grid[,c_hp_m3k_nm] = 999999


        for(i in 1:nrow(params_grid)){
          c_row = c_fold_grid[i,]
          c_lr = c_row$learning_rate
          c_epochs = c_row$epochs
          c_agr_str = c_row$agrreement_strength_lasso
          # match
          params_idx = which(params_grid$learning_rate == c_lr)
          params_idx = intersect(params_idx, which(params_grid$epochs == c_epochs))
          params_idx = intersect(params_idx, which(params_grid$agrreement_strength_lasso == c_agr_str))

          # match across-folds / only need once
          if(c_hp_fold == 1) {
            exp_idxs = which(exp_params_grid$learning_rate == c_lr)
            exp_idxs = intersect( which(exp_params_grid$epochs == c_epochs), exp_idxs)
            exp_idxs = intersect( which(exp_params_grid$agrreement_strength_lasso == c_agr_str), exp_idxs)
            params_grid[params_idx,]$tie_settler_mse  = mean(exp_params_grid[exp_idxs, ]$tie_settler_mse )
          }

          params_grid[params_idx, cns(c_fold_grid)[hf_rnk_col_idxs[c_hp_fold]]] =  c_row[,sprintf("hf%d_rnk", c_row$hyper_fold)]
          params_grid[params_idx, c_hp_m3k_nm] = c_row[, HYPERPARAM_METRIC]

        }

      }

      # compute avg rank per set
      hf_rnk_idxs = grep('hf\\d_rnk',cns(params_grid))
      params_grid$avg_rank = umap(1:nrow(params_grid), function(x) mean(as.numeric(params_grid[x,hf_rnk_idxs])))
      best_rank = min(params_grid$avg_rank)
      best_params = params_grid[params_grid$avg_rank == best_rank,]
      # best_params = params_grid[c(1,2),] # debug

      try_log_info("RANK-BASED hyperparam tuning found %d best rank results", nrow(best_params) )
      # tie-settle 1: variance
      if(nrow(best_params) > 1) {
        best_params$metric_variance = umap(1:nrow(best_params), function(x){
          sd(as.numeric(best_params[x,grep(sprintf('%s-\\d', HYPERPARAM_METRIC),cns(best_params))]), na.rm = T)
        })

        best_variance = min(best_params$metric_variance)
        best_params = best_params[best_params$metric_variance == best_variance,]
        try_log_info("Tie settler (1) hyperparam tuning found %d best result(s) based on lowest variance (%0.4f)", nrow(best_params), best_variance )
      }
      # tie-settle 2: mse
      if(nrow(best_params) > 1) {
        best_mse = min(best_params$tie_settler_mse)
        best_params = best_params[best_params$tie_settler_mse == best_mse,]
        try_log_info("Tie settler (2) hyperparam tuning found %d best result(s) based on lowest mse (%0.4f)", nrow(best_params), best_mse )
      }
      # tie-settle 3: take one at random
      if(nrow(best_params) > 1) {
        best_random = sample(1:nrow(best_params), size = 1)
        try_log_info("Tie settler (3) hyperparam tuning randomly selected element %d from %d possibilities", nrow(best_params), best_random )
        best_params = best_params[best_random,]
      }

      # set best params chosen
      training_params$learning_rate = best_params$learning_rate
      training_params$epochs =  best_params$epochs #best_params$epochs
      training_params$agrreement_strength_lasso = best_params$agrreement_strength_lasso
      training_params$verbose = verbose
      training_params$test_df = test_df

    }

  }
  else
    print("SKIP HYPERPARAM OPTIMIZATION")
  return(PRVT__train_federated_logred_model_update_hyperparams_chosen(fed_dfl, training_params))
}

PRVT__train_federated_logred_model_update_hyperparams_chosen <- function(fed_dfl, training_params) {
  # init params
  list2env(training_params, envir = environment())
  try_log_debug("@PRVT__train_federated_logreg_model_update - called with following params:")
  cat(str(training_params))

  df_all = NULL
  if(len(names(fed_dfl)) == 0)
    names(fed_dfl) = 1:len(fed_dfl)
  for(c_partition in 1:len(fed_dfl)) {
    c_df = fed_dfl[[c_partition]]
    if(len(c_df) == 0)
      next
    c_df$outcome_of_interest = c_df$outcome_of_interest
    c_df[, UOF_VN] = c_partition
    df_all = rbind(df_all, c_df)
  }
  if(!exists("test_df"))
    test_df = df_all
  # perform variable selectino here so that
  # agrreement_strength_lasso can be part of the hyperparam loop
  tmp = prep_x_matricies(df_all, test_df)
  x_train = tmp$x_train
  y_train = df_all$outcome_of_interest

  x_train[,grep(UOF_VN, cns(x_train), val = T)] = NULL
  x_train[, UOF_VN] = df_all[, UOF_VN]

  x_test = tmp$x_test
  y_test = test_df$outcome_of_interest

  x_train = as.data.frame(x_train)
  x_train$outcome_of_interest = df_all$outcome_of_interest
  x_train = data.matrix(remove_columns_with_non_full_conf_matrix(x_train, remove_almostempty_quadrants = T ,
                                                                 cutoff_thresh = MIN_N_POS_OUTCOMES_IN_DATASET))
  x_test = as.data.frame(x_test)
  x_test$outcome_of_interest = test_df$outcome_of_interest
  # variable selection
  vars_selected = try_LASSO_federated(x_train = x_train, y_train = y_train+1, agreement_strength = agrreement_strength_lasso, verbose = training_params$verbose)
  if(ONLY_VAR_SELECT)
    return(0)

  # remove intercept variable  (we know it was selected always...)
  if('(Intercept)' %in% vars_selected) {
    vars_selected = vars_selected[-which(vars_selected == '(Intercept)')]
  }

  if(len(vars_selected) == 0) {
    try_log_error("@PRVT__train_federated_logreg_model_update - vars_selected is empty" )
    return(NULL)
  }

  c_betas = rep(0, len(vars_selected)+1)
  if(!exists("test_df"))
    test_df = NULL
  if(is.null(test_df))
    test_df = df_all
  aggrgated_grad_updates = NULL
  last_coefficients = t(cbind(c("[Intercept]", vars_selected), NULL) )
  last_coefficients = last_coefficients[-1,]
  last_coefficients = as.numeric(last_coefficients)
  for(c_nm in names(fed_dfl))
    last_coefficients = rbind(last_coefficients, rep(0, len(vars_selected)+1))
  rownames(last_coefficients) = names(fed_dfl)
  colnames(last_coefficients) = c("[Intercept]", vars_selected)
  delta_coefficients = NULL
  prev_auc = NULL
  should_stop =  F
  best_auc_lambs = list()
  best_auc_lamb = -1
  # set missing columns in partitioned dfs that do exist in vars_selected to 0s
  for(c_nm in names(fed_dfl)) { # c_nm =  names(fed_dfl)[1]
    c_df = fed_dfl[[c_nm]]
    if(len(c_df) == 0)
      next
    for(c_var in vars_selected){
      if(!(c_var %in% cns(c_df))){
        c_df[,c_var] = 0
        fed_dfl[[c_nm]] = c_df
        # try_log_debug("::: missing column %s in df %s, setting to 0s", c_var, c_nm)
      }
    }
  }
  #:: Start training
  vars_per_center_to_exclude = NULL
  has_broken_counter = 0
  last_betas = c()
  df_all_prefilled = NULL
  fed_dfl_all_prefilled = list()
  achieved_good_train_cali_intercept = F
  best_distance_to_good_cali_int = 100
  for(c_epoch in 1:epochs) {
    grad_updates = NULL
    c_nms_used = c()
    # iterate over each partition dataset
    for(c_nm in names(fed_dfl)) { # c_nm =  names(fed_dfl)[1]
      c_df = fed_dfl[[c_nm]]

      if(len(c_df) == 0)
        next
      # should have both positive & negative outcome values
      if(nuniq(c_df$outcome_of_interest) != 2) {
        next
      }
      # should have at least 5 positive [or negative] outcomes
      if(howmany(c_df$outcome_of_interest == 1) < 5 || howmany(c_df$outcome_of_interest == 0) < 5) {
        next
      }

      # mark any binary variables which did not have enough [>=5] entries in each cell of their confusion matrix (to remove them from model fitting)
      c_outs = c_df$outcome_of_interest
      # only need to do this once at the start to set vars_per_center_to_exclude
      if(c_epoch == 1)
        {
        for(var_selected in vars_selected) {
          c_vals = c_df[,var_selected]
          is_binary = nuniq(c_vals) < 3
          # print(var_selected)
          if(!is_binary)
            next
          n_0s = howmany(c_vals == 0)
          n_1s = howmany(c_vals == 1)
          n_0s_and_outcome_0 = len(intersect(which(c_vals == 0), which(c_outs == 0)))
          n_0s_and_outcome_1 = len(intersect(which(c_vals == 0), which(c_outs == 1)))
          n_1s_and_outcome_0 = len(intersect(which(c_vals == 1), which(c_outs == 0)))
          n_1s_and_outcome_1 = len(intersect(which(c_vals == 1), which(c_outs == 1)))
          if(n_0s < 10 || n_1s < 10 ||
             any(c(n_0s_and_outcome_0, n_0s_and_outcome_1, n_1s_and_outcome_0, n_1s_and_outcome_1) < 5) )
          {
            vars_per_center_to_exclude = rbind(vars_per_center_to_exclude, as.data.frame(list(c_nm = c_nm, var_selected = var_selected)))
          }

        }
      }
      c_last_coefficients = last_coefficients[c_nm,]
      # get coefficient updates from lasso reg on whole current dataset
      # alpha = 0 ridge, alpha = 1 lasso ;
      coefficient_changes = NULL
      betas = NULL

      vars_to_exclude = vars_per_center_to_exclude[vars_per_center_to_exclude$c_nm == c_nm,]$var_selected
      c_vars_to_use = vars_selected
      vars_to_use_idxs = 1:len(vars_selected)
      # remove binary variables which did not have enough [>=5] entries in each cell of their confusion matrix
      if(len(vars_to_exclude) > 0)
      {
        vars_to_use_idxs = which(!c_vars_to_use %in% vars_to_exclude)
        c_vars_to_use = c_vars_to_use[vars_to_use_idxs]
        if(c_epoch == 1)
          try_log_debug("inside center %s, using %d vars", c_nm, len(c_vars_to_use))
      }
      if(len(c_vars_to_use) == 0)
        next

      if(use_ridge == F || len(vars_selected) < 3) {
        lr_fit = train_and_pool_logreg( list(c_df), vars_selected = c_vars_to_use, with_pooling = F,
                                        starting_coefficients = c_last_coefficients[c(1,vars_to_use_idxs+1 )],
                                        maxit = 1, verbose = F)[[1]]
        betas = lr_fit$coefficients

        if(any(is.na(betas)))
          betas[which(is.na(betas))] = c_last_coefficients[c(1,vars_to_use_idxs+1 )][which(is.na(betas))]
        invalid_betas_idxs = union(which(is.nan(betas)), which(abs(betas) > 300)) # when convergence fails, sometimes coefficients get set to very big values

        invalid_betas = names(betas)[invalid_betas_idxs]

        if(len(invalid_betas) > 0){
          try_log_warn("DETECTED beta > 300 or isNaN (%s = %s); [x%d]", invalid_betas, betas[invalid_betas_idxs], has_broken_counter)
          # set them to 0.5 of the previous good value
          last_coef_to_use = c_last_coefficients[c(1,vars_to_use_idxs+1 )][invalid_betas_idxs]
          try_log_debug("Setting %s to %0.3f", names(betas)[invalid_betas_idxs], last_coef_to_use)
          has_broken_counter = has_broken_counter + 1
          if(has_broken_counter > 1 && c_epoch > 1){
            try_log_warn("DETECTED beta > 300 or isNaN (%s); (x5 times!) early stopping to prevent breaking..", invalid_betas)
            should_stop = T
            betas[invalid_betas_idxs] = last_coef_to_use
            break
          }
        }
        # compute how the coefficients changed since prev iteration
        coefficient_changes = betas - c_last_coefficients[c(1,vars_to_use_idxs+1 )]
      }
      else { # using glmnet with ridge
        npos = howmany(c_df$outcome_of_interest == 1)
        nneg = howmany(c_df$outcome_of_interest == 0)
        weights_pos = sqrt(nneg/npos) #
        weights_neg = sqrt(npos/nneg)
        weights_pos = 10
        weights_neg = 1
        pos_idxs = which(c_df$outcome_of_interest == 1)
        neg_idxs = which(c_df$outcome_of_interest == 0)
        wghts = rep(weights_neg, nrow(c_df))
        wghts[pos_idxs] = weights_pos
        cv.glmnet_fit = glmnet::cv.glmnet(as.matrix(c_df[,c_vars_to_use]), c_df$outcome_of_interest,
                                          family = 'binomial', maxit = 20000, intercept = T,
                                          beta = c_last_coefficients[c(1,vars_to_use_idxs+1)],
                                          alpha = 0, nfolds = n_folds_cv_glmnet,
                                          weights = wghts )
        best_auc_lamb = (cv.glmnet_fit$lambda.1se + cv.glmnet_fit$lambda.min)/2
        # run ridge
        glmnet_fit = glmnet::glmnet(as.matrix(c_df[,c_vars_to_use]), c_df$outcome_of_interest,
                                    family = 'binomial', maxit = 5000, beta = c_last_coefficients[c(1,vars_to_use_idxs+1 )],
                                    intercept = T, alpha = 0,
                                    lambda = best_auc_lamb,
                                    weights = wghts)

        ridge_betas = coef(glmnet_fit, s = best_auc_lamb)
        ridge_betas = t(as.data.frame(as.matrix(ridge_betas)))[1,]
        betas =  ridge_betas
        coefficient_changes = betas - c_last_coefficients[c(1,vars_to_use_idxs+1 )]
      }
      # apply only fraction of the change as the new coefficients (learning rate)..
      c_multiplier = learning_rate
      coefficient_changes = coefficient_changes*c_multiplier # alpha
      # set the new coefficients (-1 to not set the intercept )
      last_coefficients[c_nm, 1] = last_coefficients[c_nm, 1] + coefficient_changes[1]
      for(var_idx in 1:len(vars_selected)) {
        if(var_idx %in% vars_to_use_idxs)
          last_coefficients[c_nm, var_idx+1] = last_coefficients[c_nm, var_idx+1] + coefficient_changes[var_idx+1]
        else
          last_coefficients[c_nm, var_idx+1] = NA # we dont care about these coefficients, they should be ignored and never used to aggregate
      }
      c_nms_used = c(c_nms_used, c_nm)
      # re-create the full vars_selected list of coefficient_changes
      full_coefficient_changes = coefficient_changes[1]

      for(i in 1:len(vars_selected)) {
        c_var = vars_selected[i]
        if(c_var %in% c_vars_to_use)
          full_coefficient_changes = c(full_coefficient_changes, coefficient_changes[which(names(coefficient_changes) == c_var)] )
        else
          full_coefficient_changes = c(full_coefficient_changes, NA )
      }
      names(full_coefficient_changes) = c(names(full_coefficient_changes)[1], vars_selected)
      # remember these updates for each partitioned dataset
      grad_updates = rbind(grad_updates, full_coefficient_changes )

    }
    if(should_stop)
      break

    ##aggregate gradients/coefficients,  FedAvg = gradient * (n_hospital_i/n_total)
    aggrgated_grad_updates = umap(cns(grad_updates), function(c_varname) {
      if(all(is.na(grad_updates[, c_varname]))) # if no center used this variable, set its beta to 0
        return(0)
      for(dataset_nm_indx in 1:len(c_nms_used)){
        dataset_nm = c_nms_used[dataset_nm_indx]
        c_df =  fed_dfl[[dataset_nm]]
        c_df = autofill_missing_var_columns(c_df, vars_selected)
        c_grad_update = grad_updates[dataset_nm_indx, c_varname]
        if(is.na(c_grad_update)) {
          grad_updates[dataset_nm_indx, c_varname] = 0
          c_grad_update = 0
        }
        df_all_with_cvar_used = df_all
        if(c_varname != "(Intercept)") # only intercept will be guaranteed to be a used variable in all partitions
          # c_varname should be used by the centers we include in df_all
          # if a varname appears in vars_per_center_to_exclude, then it should be excluded, i.e. not used
          # therefore we want the centers that the variable is not excluded from
          df_all_with_cvar_used = df_all[!df_all[, UOF_VN] %in% vars_per_center_to_exclude[vars_per_center_to_exclude$var_selected == c_varname,]$c_nm, ]
        # weighted gradient update, weighted by size of dataset
        if(aggr_technique == "fedavg")
          grad_updates[dataset_nm_indx, c_varname] = c_grad_update * (nrow(c_df)/ nrow(df_all_with_cvar_used))
        # plain average
        if(aggr_technique == "average")
          grad_updates[dataset_nm_indx, c_varname] = c_grad_update / len(fed_dfl)
      }
      return(sum(grad_updates[,c_varname]))
    }  )
    names(aggrgated_grad_updates) = cns(grad_updates)

    delta_coefficients = c_betas - aggrgated_grad_updates

    last_betas = c_betas
    c_betas = c_betas + aggrgated_grad_updates
    bnms = names(c_betas)
    names(c_betas) = bnms

    ## compute training calibration here and see if you can detect a good time to stop
    if(c_epoch == 1) { # only need to do this once
      df_all_prefilled = autofill_missing_var_columns(df_all, vars_selected)
      for(c_fdl in 1:len(fed_dfl)) {
        if(!is.null(fed_dfl[[c_fdl]]))
          fed_dfl_all_prefilled[[c_fdl]] = autofill_missing_var_columns(fed_dfl[[c_fdl]], vars_selected)
        else
          fed_dfl_all_prefilled[[c_fdl]] = NULL
      }
    }

    # set the current coefficients of each partitioned dataset to the new aggregated ones
    #-1 to not set the intercept
    for(ii in 1:nrow(last_coefficients))
      last_coefficients[ii,] = c_betas

    # only start to look at training cali after some epochs (we know intercept will keep improving until then at least..)
    if(c_epoch > 1 && c_epoch %% 3 == 0) {
      train_preds = PRVT__try_predict_model_updates(c_betas, df_all_prefilled[,vars_selected])
      train_cali = try_get_calibration(train_preds, df_all$outcome_of_interest, verbose = F)
      is_curr_cal_int_good = train_cali$ci.cal.intercept[1] <= 0 && train_cali$ci.cal.intercept[2] >= 0
      c_dist_to_cal_int = if(is.numeric(train_cali$cal.intercept)) abs(train_cali$cal.intercept) else 999
      try_log_debug("train cali int %0.2f [%0.2f : %0.2f], slope %0.2f [%0.2f : %0.2f] ",
                    train_cali$cal.intercept, train_cali$ci.cal.intercept[1], train_cali$ci.cal.intercept[2],
                    train_cali$cal.slope, train_cali$ci.cal.slope[1], train_cali$ci.cal.slope[2])
      if(best_distance_to_good_cali_int > c_dist_to_cal_int) { # we are improving in cali int still here
        best_distance_to_good_cali_int = c_dist_to_cal_int
      }
      is_curr_cal_int_good = if(is.na(is_curr_cal_int_good)) F else is_curr_cal_int_good
      if(!achieved_good_train_cali_intercept && is_curr_cal_int_good){
        achieved_good_train_cali_intercept = T # you reached good cali, maybe stop here even, or wait until you lose it?
        try_log_info("********************************************* achieved_good_train_cali_intercept! ")
        # break
      }

    }

    # log and plot intermediate results
    if(!is.null(test_df) && verbose) {
      if(c_epoch == 1) # only need to do this once
        test_df = autofill_missing_var_columns(test_df, vars_selected)
      if(c_epoch %% DEBUG_ON_EVERY_N_EPOCHS  == 0 ) {
        print("debug")
        print("DEBUG_ON_EVERY_N_EPOCHS set breakpoint here")
        print("debug")
      }
      test_preds = PRVT__try_predict_model_updates(c_betas, test_df[,vars_selected])
      y_test = as.numeric(test_df$outcome_of_interest)
      sq_err = sqrt_err(betas = c_betas, x = test_df[,vars_selected],  y = y_test)
      test_auc = try_get_aucroc_ci(predictions = test_preds, true_vals = y_test)
      prev_auc = ifelse(c_epoch == 1, test_auc[2] - 1, prev_auc)
      delta_auc = abs(prev_auc - test_auc[2])
      prev_auc = test_auc[2]
      try_log_trace("Delta AUC - %0.5f", delta_auc)
      early_stop_delta = 0
      if(delta_auc < early_stop_delta) {
        try_log_info("Stopped after %d epochs as AUC delta [%0.5f] was smaller than threshold %0.5f",
                     c_epoch , delta_auc, early_stop_delta)
        should_stop =  T
      }
      if(verbose) {
        try_log_debug("E%d TSTAUC=%0.4f; asb=%0.3f; MSE=%0.3f; nnzs=%d; sprd=%0.3f",
                      c_epoch, test_auc[2], sum(abs(c_betas[-1])), mean(sq_err), howmany(abs(c_betas)> 0), try_get_spread_metric(test_preds)*10)
      }

      if(plot_hist_every_n_epochs > 0 && numbers::mod(c_epoch, plot_hist_every_n_epochs) == 0 && howmany(abs(c_betas)> 0) > 1) {

        tryCatch({
          par(mfrow=c(2,2))

          cal_reses = try_get_calibration(test_preds,  y_test, verbose = F)
          if(len(cal_reses$ci.cal.intercept) != 0) {
            txt1 = sprintf("CAL.int.95 %0.2f %0.2f", cal_reses$ci.cal.intercept[1], cal_reses$ci.cal.intercept[2])
            txt2 = sprintf("CAL.slope.95 %0.2f %0.2f", cal_reses$ci.cal.slope[1], cal_reses$ci.cal.slope[2])
            plot.new()
            text(1,0.5, txt1, cex=1, pos=2)
            text(1,0.8, txt2, cex=1, pos=2)

          }
          h_plot = hist(test_preds, breaks = 1000, main = sprintf("fedavg hist of test predictions (epoch %d)", c_epoch))
          text_x_pos = min(h_plot$breaks) + ((max(h_plot$breaks) - min(h_plot$breaks))*0.4)
          text_y_pos = max(h_plot$counts)*0.9
          barplot(c(c_betas[1]/10,c_betas[-1]), las = 2)
          text(text_x_pos, text_y_pos, sprintf("AUC = %0.4f (95%%CI %0.2f-%0.2f)", test_auc[2], test_auc[1], test_auc[3] ), col = "red")
          plot_calibration_graph(test_preds, test_df$outcome_of_interest, smoothed = F)
        }, error = function(e) { try_log_error("Error calling plot_calibration_graph ")}, finally = {})
      }
      if(should_stop)
        break
    }
  }  # end of epoch training loop
  if(should_stop) {
    c_betas = last_betas  # use the last good betas before breaking
    # c_betas = betas
  }

  intercept = 0
  slope = 1
  # perform l1 recallibration
  if(FEDAVG_L1_RECALL_STRATEGY == 'FEDERATED') {
    n_nn = 0
    for(i in 1:len(fed_dfl))
      if(!is.null(fed_dfl[[i]]))
        n_nn = n_nn+1
    recall_models = rep(0, n_nn)
    n_nct = 0
    recal_ms = list()
    for(i in 1:len(fed_dfl)) {
      c_df = fed_dfl_all_prefilled[[i]]
      if(is.null(c_df))
        next
      n_nct = n_nct+1
      # todo: implement l1 recalibration
      c_betas_clone = c_betas
      c_preds = PRVT__try_predict_model_updates(c_betas_clone, c_df[,vars_selected])
      c_outs = c_df$outcome_of_interest
      recal_res = perform_L1_recalibration(c_preds, c_outs)
      recal_fn = recal_res$recal_pred_fn
      recal_m = recal_res$recal_model
      recal_ms[[i]] = recal_m

    }
    pooled_res <- mice::pool(recal_ms)$pooled
    intercept = pooled_res$estimate[1]
    slope = pooled_res$estimate[2]
  }
  if(FEDAVG_L1_RECALL_STRATEGY == 'GLOBAL') {
    df_global_prefilled = NULL
    for(i in 1:len(fed_dfl)) {
      c_df = fed_dfl_all_prefilled[[i]]
      df_global_prefilled = rbind(df_global_prefilled, c_df)
    }
    c_betas_clone = c_betas # why do we clone?
    c_preds = PRVT__try_predict_model_updates(c_betas_clone, df_global_prefilled[,vars_selected])
    c_outs = df_global_prefilled$outcome_of_interest
    recal_res = perform_L1_recalibration(c_preds, c_outs)
    recal_fn = recal_res$recal_pred_fn
    recal_m = recal_res$recal_model
    intercept = recal_m$coefficients[1]
    slope = recal_m$coefficients[2]
  }

  pooled_recal_predict_fn <- function(new_preds) {
    # Calibrate the probabilities using the logistic regression equation
    calibrated_preds <- plogis(intercept + slope * new_preds)
    return(calibrated_preds)
  }

  # # fedavg of best intercepts
  # multipliers_volume = multipliers_volume/ sum(multipliers_volume)
  # good_avg_intercept = sum(improved_beta1s*multipliers_volume)
  nonzero_betas = c()
  if(any(c_betas != 0))
    nonzero_betas = c_betas[which(c_betas != 0)]
  else {
    try_log_warn("Model produced only zero coefficients! returning empty model...")
    return(NULL)
  }

  training_params$vars_selected = names(nonzero_betas)
  ret_recal_fn = if(FEDAVG_PERFORM_L1_RECALL) pooled_recal_predict_fn else NULL
  r_obj = list(coefficients = nonzero_betas, training_params = training_params, recal_fn = ret_recal_fn)
  class(r_obj) = FEDERATION_MECHANISM_UPDATE_SHARING
  return(r_obj)
}

# returns recalibration funciton
perform_L1_recalibration <- function(preds, outs) {
  # Create a calibration dataset
  calibration_data <- data.frame(preds = preds, outs = outs)

  # Train a logistic regression model on the calibration dataset
  model <-glm(outs ~ preds, data = calibration_data, family = binomial())

  # Extract the coefficients of the trained model
  intercept <- coef(model)[1]
  slope <- coef(model)[2]

  recal_predict_fn <- function(new_preds) {
    # Calibrate the probabilities using the logistic regression equation
    calibrated_preds <- plogis(intercept + slope * new_preds)
    return(calibrated_preds)
  }

  return(list( recal_model = model,
               recal_pred_fn = recal_predict_fn))
}



