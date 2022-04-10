import os

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
CLASS_DF_PATH = os.path.abspath(os.path.join(TEST_PATH, "titanic.csv"))
CLASS_CAT_COLS = ["Sex", "Cabin"]
CLASS_CONT_COLS = ["Age", "Fare"]
CLASS_DEP_VAR = "Survived"
TRAIN_SPLIT = 0.8
STRATIFY_ERROR_MARGIN = 0.2