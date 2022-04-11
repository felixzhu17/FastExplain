import os

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
CLASS_DF_PATH = os.path.abspath(os.path.join(TEST_PATH, "titanic.csv"))
CAT_COLS = ["Sex", "Cabin"]
CONT_COLS = ["Age", "Fare"]
CLASS_DEP_VAR = "Survived"
REG_DEP_VAR = "SibSp"
TRAIN_SPLIT = 0.8
STRATIFY_ERROR_MARGIN = 0.02
