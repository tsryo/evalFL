# Author: T.Y.
# Date created: 15.06.2022

# set wd to top-level-dir of current project
# setwd(sprintf("%s/..",dirname(rstudioapi::getActiveDocumentContext()$path)))


# Script with common imports to call (easier to then source this script instead of all other scripts separately)

source("../commons/src/orchestrator.R")
source("src/fl_constants.R")
source("src/fl_util.R")
source("src/gradient_update_model.R")
source("src/fed_feature_selection.R")

# DATA PARTITIONING =  Horizontal
# ML MODEL = Linear/Logistic
# PRIVACY MECHANISM = TBD
# COMMUNICAITON ARCHITECTURE = Centralized
# UNIT OF FEDERATION =  hospital
# MOTIVAITON OF FEDERATION =  Incentive? (future regulation?)


























