from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class CustomRidge(BaseEstimator):
    """
    Provides two models: main model for normal cases and outlier-specific model
    """
    def __init__(self):
        self.main = Ridge(alpha=3e-2) 
        self.outliers = Ridge(alpha=3e-1)

        self.main_scaler = RobustScaler()
        self.outliers_scaler = RobustScaler()


    def fit(self, X, y):
        groups = X['outliers']
        X = X.drop(columns=['outliers'])

        main_mask = (groups == 0).values
        main_mask[X.loc[:, 'very_bad'] == True] = 0
        
        X_main = self.main_scaler.fit_transform(X[main_mask])

        self.main.fit(self.main_scaler.transform(X[main_mask]), y[main_mask])
        self.outliers.fit(self.outliers_scaler.fit_transform(X), y)     
        self.pred_shape = y.shape[-1]

        return self

    def predict(self, X):
        groups = X['outliers']
        X = X.drop(columns=['outliers'])
        
        predictions = np.zeros((X.shape[0], self.pred_shape))

        main_mask = (groups == 0).values
        if main_mask.sum():
            predictions[main_mask] = self.main.predict(self.main_scaler.transform(X[main_mask]))                                                 
        if (~main_mask).sum():
            predictions[~main_mask] = self.outliers.predict(self.outliers_scaler.transform(X[~main_mask]))
        
        return predictions


model = CustomRidge()

oof_pred = cross_val_predict(model, train, train_labels.values, cv=100)

print(f"# R2 score: {r2_score(train_labels, oof_pred):.3f}")

sigma_pred = mean_squared_error(train_labels, oof_pred, squared=False)
  
print(f"# Root mean squared error: {sigma_pred:.6f}")

col = 1
plt.scatter(oof_pred[:,col], train_labels.iloc[:,col], s=15, c='lightgreen')
plt.gca().set_aspect('equal')
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.title('Comparing y_true and y_pred')
plt.show()

# clipping
oof_pred = np.maximum(oof_pred, 0.003)
oof_pred = np.minimum(oof_pred, 0.1)



class SigmaPredictor:
    """
    Class for sigma predicting
    """
    def __init__(self):
        self.sigmas = {}
        
    def fit(self, y_pred, y_true, outliers, very_bad):        
        outliers = [i for i in outliers if i not in very_bad]

        self.sigmas['outliers'] = self._calc(y_pred[outliers], y_true[outliers]) * 5

        main = self._del_outliers(np.ones(len(y_pred), dtype=bool), outliers + list(very_bad))
        self.sigmas['main'] = self._calc(y_pred[main], y_true[main])

        print({ k: v.mean() for k, v in self.sigmas.items() })

    def predict(self, sigma_pred, y_pred, outliers, very_bad, bootstrap_preds=None):
        if len(outliers) > 0:
            sigma_pred[outliers] = self.sigmas['outliers']

        main = self._del_outliers(np.ones(len(y_pred), dtype=bool), outliers)
        if main.sum() > 0:
            sigma_pred[main] = self.sigmas['main']

        W1 = 0.75
        W2 = 1.0 - W1
        
        sigma_pred[main, :] = bootstrap_preds[main, :] * W1 + sigma_pred[main, :] * W2
        sigma_pred[outliers] = bootstrap_preds[outliers] * 1.5

        sigma_pred[very_bad] = 0.003 
        for i in very_bad:
            if i in outliers:
                continue
            sigma_pred[i, :] = 0.5 * bootstrap_preds[i, :] + 0.5 * sigma_pred[i, :]

        return sigma_pred
        

    def _calc(self, y_pred, y_true): # calculate rmse for each frequency
        sigmas = []
        for i in range(y_pred.shape[1]):
            sigmas.append(mean_squared_error(y_pred[:, i], y_true[:, i], squared=False))
        return np.array(sigmas)

    def _del_outliers(self, mask, outliers):
        for i in range(len(mask)):
            if i in outliers:
                mask[i] = False
        return mask                         


def postprocessing(pred_array, index, sigma_pred, sigma_predictor, outliers, very_bad, bootstrap_preds=None, column_names=None):
    """
    Creates a submission DataFrame with mean predictions and uncertainties.

    Parameters:
    - pred_array: ndarray of shape (n_samples, 283)
    - index: pandas.Index of length n_samples
    - sigma_pred: float or ndarray of shape (n_samples, 283)
    - column_names: list of wavelength column names (optional)

    Returns:
    - df: DataFrame of shape (n_samples, 566)
    """
    n_samples, n_waves = pred_array.shape

    if column_names is None:
        column_names = [f"wl_{i+1}" for i in range(n_waves)]
    
    sigma_pred = sigma_predictor.predict(np.zeros_like(pred_array), pred_array, outliers, very_bad, bootstrap_preds=bootstrap_preds)

    # Safety check
    assert sigma_pred.shape == pred_array.shape, "sigma_pred must match shape of pred_array"
    assert len(index) == n_samples, "Index length must match number of rows"

    df_mean = pd.DataFrame(pred_array.clip(0, None), index=index, columns=column_names)
    df_sigma = pd.DataFrame(sigma_pred, index=index, columns=[f"sigma_{i+1}" for i in range(n_waves)])

    return pd.concat([df_mean, df_sigma], axis=1)


model.fit(train, train_labels)

sigma_predictor = SigmaPredictor()
very_bad = np.arange(train_labels.shape[0])[train['very_bad'].values == True]
sigma_predictor.fit(oof_pred, train_labels.values, outliers, very_bad)

def bootstrap_uncertainty_inference( 
    X_train,
    y_train,
    X_test,
    n_bootstraps = 100, 
    random_state = 42,
):
    """
    Sigma estimation via bootstrapping
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    y_train_values = y_train
        
    n_test_samples = X_test.shape[0]
    n_targets = y_train_values.shape[1]
    
    predictions = np.full((n_test_samples, n_targets, n_bootstraps), np.nan)
    
    bootstrap_iter = range(n_bootstraps)
    bootstrap_iter = tqdm(bootstrap_iter, desc="bootstrap interations")
    
    for b in bootstrap_iter:
        bootstrap_indices = np.random.choice(
            len(X_train), size=len(X_train), replace=True
        )
        
        X_bootstrap = X_train.iloc[bootstrap_indices].reset_index(drop=True)
        y_bootstrap = y_train_values.iloc[bootstrap_indices].reset_index(drop=True)
        
        model_bootstrap = CustomRidge()
        model_bootstrap.fit(X_bootstrap, y_bootstrap)
        
        y_pred = model_bootstrap.predict(X_test)   
        predictions[:, :, b] = y_pred
     
    return predictions

import pickle
import gc


test_adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/test_star_info.csv', index_col='planet_id')
sample_submission = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/sample_submission.csv', index_col='planet_id')
wavelengths = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')
test_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/test_star_info.csv')

del data_train

gc.collect()
os.environ["PREPROCESS_MODE"] = "test"

!python preprocess.py
!rm -rf *AIRS-CH0_signal*

data_test = np.load(f"signal_{VERSION}.npy")

outliers, test_features = feature_engineering(test_star_info, data_test)
outliers = np.arange(test_features.shape[0])[outliers]
very_bad = np.arange(test_features.shape[0])[test_features['very_bad'].values == True]

test_pred = model.predict(test_features)

boot_pred = bootstrap_uncertainty_inference(train, train_labels, test_features, n_bootstraps=1000)
test_bootstrap_preds = boot_pred.std(-1) * 2.75

test_pred = np.maximum(test_pred, 0.003)
test_pred = np.minimum(test_pred, 0.1)

print('sigma', sigma_pred)


def postprocessing(pred_array, index, sigma_pred, sigma_predictor, outliers, very_bad, bootstrap_preds=None, column_names=None):
    """
    Convert predictions and uncertainty into final submission DataFrame
    """

    sigma_array = sigma_predictor.predict(np.zeros_like(pred_array), pred_array, outliers, very_bad, bootstrap_preds=bootstrap_preds)

    df_pred = pd.DataFrame(pred_array.clip(0, None), index=index, columns=column_names)
    df_sigma = pd.DataFrame(sigma_array, index=index, columns=[f"sigma_{i}" for i in range(1, len(column_names)+1)])
    return pd.concat([df_pred, df_sigma], axis=1)


submission_df = postprocessing(
    pred_array=test_pred,
    index=sample_submission.index,
    sigma_pred=sigma_pred,
    sigma_predictor=sigma_predictor,
    column_names=wavelengths.columns,
    bootstrap_preds=test_bootstrap_preds,
    outliers=outliers,
    very_bad=very_bad
)

submission_df.to_csv('submission.csv')