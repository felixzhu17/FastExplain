# REG_MODELS = {"rf": Models().rf_reg, "xgb": Models().xgb_reg, "ebm": Models().ebm_reg}
# CLASS_MODELS = {
#     "rf": Models().rf_class,
#     "xgb": Models().xgb_class,
#     "ebm": Models().ebm_class,
# }


# class ModelHelper(Models, Clustering, Explain, Metrics, Sensitivity, DataPrep):
#     def model_data(
#         self,
#         df,
#         cat_names,
#         cont_names,
#         dep_var,
#         type="reg",
#         *args,
#         **kwargs,
#     ):
#         if type == "reg":
#             return Regression(
#                 df,
#                 cat_names,
#                 cont_names,
#                 dep_var,
#                 *args,
#                 **kwargs,
#             )
#         elif type == "class":
#             return Classification(
#                 df,
#                 cat_names,
#                 cont_names,
#                 dep_var,
#                 *args,
#                 **kwargs,
#             )
#         else:
#             raise ValueError("Type can only be reg or class")


# class Regression(
#     Models,
#     ClusteringClassified,
#     ExplainClassified,
#     Metrics,
#     SensitivityClassified,
#     DataPrep,
# ):
#     def __init__(
#         self,
#         df,
#         cat_names,
#         cont_names,
#         dep_var,
#         model="rf",
#         perc_train=0.8,
#         seed=0,
#         splits=None,
#         normalize=False,
#         one_hot=False,
#         fill_strategy="median",
#         fill_const=0,
#         na_dummy=True,
#         reduce_memory=True,
#         *args,
#         **kwargs,
#     ):

#         self.data = self.prepare_data(
#             df,
#             cat_names=cat_names,
#             cont_names=cont_names,
#             dep_var=dep_var,
#             return_class=True,
#             perc_train=perc_train,
#             seed=seed,
#             splits=splits,
#             normalize=normalize,
#             one_hot=one_hot,
#             fill_strategy=fill_strategy,
#             fill_const=fill_const,
#             na_dummy=na_dummy,
#             reduce_memory=reduce_memory,
#         )

#         self.benchmark = self.data.train_y.mean()
#         self.benchmark_rmse = self.r_mse(self.benchmark, self.data.train_y), self.r_mse(
#             self.benchmark, self.data.val_y
#         )

#         if model in REG_MODELS:
#             self.m = REG_MODELS[model](
#                 self.data.train_xs, self.data.train_y, *args, **kwargs
#             )
#         else:
#             raise ValueError(f"Model can only be one of {', '. join(REG_MODELS)}")

#         self.model_rmse = self.m_rmse(
#             self.m, self.data.train_xs, self.data.train_y
#         ), self.m_rmse(self.m, self.data.val_xs, self.data.val_y)

#         ExplainClassified.__init__(
#             self, self.m, self.data.xs, self.data.df, self.data.dep_var
#         )
#         ClusteringClassified.__init__(self, self.m, self.data.xs)
#         SensitivityClassified.__init__(self, self.m, self.data.xs)


# class Classification(
#     Models,
#     ExplainClassified,
#     ClusteringClassified,
#     Metrics,
#     SensitivityClassified,
#     DataPrep,
# ):
#     def __init__(
#         self,
#         df,
#         cat_names,
#         cont_names,
#         dep_var,
#         model="rf",
#         perc_train=0.8,
#         seed=0,
#         splits=None,
#         normalize=False,
#         one_hot=False,
#         fill_strategy="median",
#         fill_const=0,
#         na_dummy=True,
#         reduce_memory=True,
#         *args,
#         **kwargs,
#     ):

#         self.data = self.prepare_data(
#             df,
#             cat_names=cat_names,
#             cont_names=cont_names,
#             dep_var=dep_var,
#             perc_train=perc_train,
#             seed=seed,
#             splits=splits,
#             normalize=normalize,
#             one_hot=one_hot,
#             fill_strategy=fill_strategy,
#             fill_const=fill_const,
#             na_dummy=na_dummy,
#             reduce_memory=reduce_memory,
#             return_class=True,
#         )

#         if model in CLASS_MODELS:

#             self.m = CLASS_MODELS[model](
#                 self.data.train_xs, self.data.train_y, *args, **kwargs
#             )
#         else:
#             raise ValueError(f"Model can only be one of {', '. join(CLASS_MODELS)}")

#         ExplainClassified.__init__(
#             self, self.m, self.data.xs, self.data.df, self.data.dep_var
#         )
#         ClusteringClassified.__init__(self, self.m, self.data.xs)
#         SensitivityClassified.__init__(self, self.m, self.data.xs)

#     def plot_roc_curve(self, *args, **kwargs):
#         return Metrics().plot_roc_curve(
#             self.m, self.data.val_xs, self.data.val_y, *args, **kwargs
#         )

#     def confusion_matrix(self, *args, **kwargs):
#         return Metrics().confusion_matrix(
#             self.m, self.data.val_xs, self.data.val_y, *args, **kwargs
#         )
