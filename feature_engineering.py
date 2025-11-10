from scipy.signal import savgol_filter
import warnings
from scipy.stats import kurtosis, skew
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares, minimize
from sklearn.metrics import mean_squared_error
from scipy.ndimage import median_filter


warnings.simplefilter("ignore")

A_BINNING = 15

# threshold values for identifying outliers
bad_low = 20
bad_up = 354

buf = 15  # for lower1, upper1
buf1 = 10  # for lower, mid1, mid2, upper


def calculate_weights(a, lower, mid1, mid2, upper, lower1, upper1, is_outlier=False):
    """
    Calculating weights for averaging based on SNR
    """
    max_len = a.shape[0] - 1
    if is_outlier:
        return np.ones_like(a[0]) / a.shape[-1]
    else:
        y_combined = np.concatenate([a[:max(lower - buf1, 1), :], a[min(upper + buf1, max_len):, :]], axis=0)
        ratio = y_combined.mean(0) / y_combined.std(0)
        return ratio / ratio.sum()


def calc_for_outliers(a, lower, upper):
    """
    Estimating transit depth for outlier cases
    """
    max_len = len(a) - 1
            
    if lower + buf < upper - buf:
        obs = a[lower + buf : upper - buf].mean()
    else:
        obs = a[lower : upper].mean()

    if lower - buf >= 10 and max_len - upper - buf >= 10:
        unobs = (np.median(a[:max(lower - buf, 1)]) + np.median(a[min(upper + buf, max_len):])) / 2
    elif lower >= max_len - upper:
        unobs = np.median(a[:max(lower - buf, 1)])
    else:
        unobs = np.median(a[min(upper + buf, max_len):])

    arr1 = 1 - (obs / unobs)
    arr2 = 1 - a[(lower + upper) // 2] / unobs

    return arr1, arr2


def calc_err(x_combined, y_combined, degree):
    """
    Calculating the error to obtain the optimal polynomial degree
    """
    max_len = 374 # hardcoded for BINNING=15
    
    poly_guess = np.polyfit(x_combined, y_combined, degree)
    inter = np.polyval(poly_guess, np.arange(max_len + 1))
    err = mean_squared_error(y_combined, inter[x_combined], squared=False)

    # penalizing rmse, high polynomial degree and small number of points in curve fitting.
    return err * degree**(1 - len(x_combined) / max_len)

    
def calc_depth_and_detrend(a, lower, mid1, mid2, upper, lower1, upper1, is_outlier=False, unstable=False, fixed_degree=None):
    """
    Main function for transit depth estimation and detrending
    
    Parameters:
        - a: 1d numpy array of observation points
        - lower, mid1, mid2, upper, lower1, upper1: transit boundary points
        - is_outlier: boolean flag indicating if the data point is an outlier
        - unstable: if True, don't use curve fitting
        - fixed_degree: if not None, use provided degree; otherwise, find optimal degree
    
    Returns:
        - (arr1, arr2): tuple containing the averaged transit depth and the transit depth at mid-transit
    """
    max_len = len(a) - 1
    degree = 3
    
    if is_outlier:
        a /= a.mean()
        arr1, arr2 = calc_for_outliers(a, lower1, upper1)
    else:
        a /= a.mean()

        # region outside the transit
        x_combined = np.concatenate([np.arange(max(lower - buf1, 1)), np.arange(min(upper + buf1, max_len), max_len + 1)], axis=-1)
        y_combined = a[x_combined]
            
        if fixed_degree is None: # find the optimal degre
            best_val = 10**100
            for j in [1, 2, 3, 4, 5]:
                if calc_err(x_combined, y_combined, j) < best_val:
                    best_val = calc_err(x_combined, y_combined, j)
                    degree = j
        else:
            degree = fixed_degree

        obs = a[mid1 : mid2]
        
        if unstable: # no curve fitting
            unobs = y_combined.mean()
        else:
            poly_guess = np.polyfit(x_combined, y_combined, degree)         
            inter = np.polyval(poly_guess, np.arange(max_len + 1))
                
            a /= inter
            inter /= inter
            unobs = inter[mid1 : mid2]

        arr1 = 1 - np.mean(obs / unobs)
        arr2 = 1 - a[(lower1 + upper1) // 2] / unobs.mean()
                

    if np.isnan(arr1):
        arr1 = 0
    if np.isnan(arr2):
        arr2 = 0
        
    return arr1, arr2


def calc_slope(a, lower, mid1, mid2, upper, lower1, upper1, is_outlier=False):
    """
    Calculate transit wall steepness as slope between contact points
    """
    max_len = len(a) - 1
    
    if not (lower < mid1 < mid2 < upper) or is_outlier:
        return 0
    else:
        return ((a[mid1] - a[lower]) / (mid1 - lower) - (a[upper] - a[mid2]) / (upper - mid2)) / 2


def calc_slope_2(a, lower, mid1, mid2, upper, lower1, upper1, is_outlier=False):
    """
    Calculate transit bottom curvature as slope between mid-transit 
    and contact point
    """
    max_len = len(a) - 1
    
    if not (lower < mid1 < mid2 < upper) or is_outlier:
        return 0
    else:
        mid_ind = (mid1 + mid2) // 2
        if not (mid1 < mid_ind < mid2):
            return 0
        return ((a[mid_ind] - a[mid1]) / (mid_ind - mid1) - (a[mid2] - a[mid_ind]) / (mid2 - mid_ind)) / 2


# calcluating slopes for unsimmetric cases
def calc_slope_2_left(a, lower, mid1, mid2, upper, lower1, upper1, is_outlier=False):
    max_len = len(a) - 1
    
    if not (lower < mid1 < mid2 < upper):
        return 0
    else:
        mid_ind = (mid1 + mid2) // 2
        if not (mid1 < mid_ind < mid2):
            return 0
        return (a[mid_ind] - a[mid1]) / (mid_ind - mid1)


def calc_slope_2_right(a, lower, mid1, mid2, upper, lower1, upper1, is_outlier=False):
    max_len = len(a) - 1
    
    if not (lower < mid1 < mid2 < upper):
        return 0
    else:
        mid_ind = (mid1 + mid2) // 2
        if not (mid1 < mid_ind < mid2):
            return 0
        return (a[mid2] - a[mid_ind]) / (mid2 - mid_ind)


# gradient slopes
def calc_curv_left(a, lower, mid1, mid2, upper, lower1, upper1, is_outlier=False):
    if not (lower < mid1 < mid2 < upper) or is_outlier:
        return 0
    else:
        mid_ind = (mid1 + mid2) // 2
        a = savgol_filter(np.gradient(a), 41, 1)
        return (a[mid_ind] - a[mid1]) / (mid_ind - mid1)


def calc_curv_right(a, lower, mid1, mid2, upper, lower1, upper1, is_outlier=False):
    if not (lower < mid1 < mid2 < upper) or is_outlier:
        return 0
    else:
        mid_ind = (mid1 + mid2) // 2
        a = savgol_filter(np.gradient(a), 41, 1)
        return (a[mid2] - a[mid_ind]) / (mid2 - mid_ind)


def calc_perc(a, lower, upper, q, is_outlier=False):
    """
    Calculate the percentile of the transit depth, assumes the input signal is already detrended
    """
    if is_outlier:
        return 0
    return np.quantile(1 - a[lower : upper], q)
    

def feature_engineering(star_info, data):
    """
    Prepares features for training or inference
    
    Parameters:
        - star_info: star metadata DataFrame
        - data: 3d data array (samples, time, frequencies)
    
    Returns:
        tuple of DataFrame and outliers mask
    """
    df = pd.DataFrame()

    cut_inf, cut_sup = 36, 318
    
    signal = np.concatenate(
        [data[:, :, 0][:, :, None], data[:, :, cut_inf:cut_sup]], axis=2
    )
    max_len = signal.shape[1] - 1
        
    lower, mid1, mid2, upper, lower1, upper1 = get_breakpoints(signal[:, :, 1:].mean(-1))
    boundaries = (lower, mid1, mid2, upper, lower1, upper1) 

    # identifying outliers
    outliers = (np.array(lower1) < bad_low) | (np.array(upper1) > bad_up) | (np.array(lower) < bad_low) | (np.array(upper) > bad_up)
    for i in range(signal.shape[0]):
        if not (lower[i] < mid1[i] < mid2[i] < upper[i]):
            outliers[i] = 1
    
    signal_mean_raw = np.zeros(signal.shape[:2])
    for i in tqdm(range(signal_mean_raw.shape[0])): # weighted averaging along frequency dimension
        weights = calculate_weights(signal[i, :, 1:], *(b[i] for b in boundaries), outliers[i])
        signal_mean_raw[i, :] = (signal[i, :, 1:] @ weights.T).T

    signal_mean = savgol_filter(signal_mean_raw, 11, 1)

    # frequency set for precise depth estimation via curve fitting (less robust)
    good_waves = [1, 6, 11, 16, 21, 26, 31, 36, 41, 51, 61, 71, 76, 81, 86, 91, 96, 101, 106, 111, 121, 131, 141, 151, 161, 171, 196, 201, 206]
    for i in tqdm(range(len(signal_mean))):

        # filter very bad cases :) (exclude from training, use larger sigma for prediction)
        df.loc[i, 'very_bad'] = (lower1[i] < bad_low // 2 or upper1[i] > max_len - (max_len - bad_up) // 2)
        if os.environ["PREPROCESS_MODE"] == 'train' and star_info.loc[i, 'planet_id'] in [2486733311, 2554492145]:
            df.loc[i, 'very_bad'] = True


        # averaged and mid-transit depth estimation
        fake_avg, fake_mid = calc_depth_and_detrend(signal_mean[i].copy(), *(b[i] for b in boundaries), outliers[i], unstable=True)
        fake_avg_2, fake_mid_2 = calc_depth_and_detrend(signal_mean[i].copy(), *(b[i] for b in boundaries), outliers[i], fixed_degree=3)
        df.loc[i, 'average_depth'], df.loc[i, 'mid_depth'] = calc_depth_and_detrend(signal_mean[i], *(b[i] for b in boundaries), outliers[i])

        norm_coef = (1 - df.loc[i, 'average_depth']) / (1 - fake_avg)
        norm_coef_mid = (1 - df.loc[i, 'mid_depth']) / (1 - fake_mid)
        norm_coef_2 = (1 - df.loc[i, 'average_depth']) / (1 - fake_avg_2)
                        
        fg1_signal = signal[i, :, 0].copy()
        fg1_signal = savgol_filter(fg1_signal, 11, 1)
        fg1_slope = savgol_filter(signal[i, :, 0].copy(), 41, 2)
        _, _ = calc_depth_and_detrend(fg1_slope, *(b[i] for b in boundaries), outliers[i], fixed_degree=3) # for detrending
        
        df.loc[i, 'fg1_average_depth'], df.loc[i, 'fg1_mid_depth'] = calc_depth_and_detrend(fg1_signal, *(b[i] for b in boundaries), 
                                                                                            outliers[i], 
                                                                                            fixed_degree=3)

        
        # transit depth percentiles
        for q in [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            df.loc[i, f'q_1_{q}'] = calc_perc(signal_mean[i], mid1[i], mid2[i], q, outliers[i])
            df.loc[i, f'q_2_{q}'] = calc_perc(signal_mean[i], lower[i] - buf1, upper[i] + buf1, q, outliers[i])
            df.loc[i, f'q_3_{q}'] = calc_perc(signal_mean[i], lower[i], upper[i], q, outliers[i]) 
        for q in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
            df.loc[i, f'fg1_q_1_{q}'] = calc_perc(fg1_signal, mid1[i], mid2[i], q, outliers[i])
            df.loc[i, f'fg1_q_2_{q}'] = calc_perc(fg1_signal, lower[i], upper[i], q, outliers[i])
            df.loc[i, f'fg1_q_3_{q}'] = calc_perc(fg1_signal, lower[i] - buf1, upper[i] + buf1, q, outliers[i])

        
        # slope features
        df.loc[i, 'slope'] = calc_slope(signal_mean[i], *(b[i] for b in boundaries), outliers[i])
        df.loc[i, 'slope_2'] = calc_slope_2(signal_mean[i], *(b[i] for b in boundaries), outliers[i])
        df.loc[i, 'slope_2_left'] = calc_slope_2_left(signal_mean[i], *(b[i] for b in boundaries), outliers[i])
        df.loc[i, 'slope_2_right'] = calc_slope_2_right(signal_mean[i], *(b[i] for b in boundaries), outliers[i])
        df.loc[i, 'slope_g'] = max(0, -df.loc[i, 'slope_2'])**0.5
                          
        df.loc[i, 'fg1_slope'] = calc_slope(fg1_slope, *(b[i] for b in boundaries), outliers[i])     
        df.loc[i, 'fg1_slope_2'] = calc_slope_2(fg1_slope, *(b[i] for b in boundaries), outliers[i])      
        df.loc[i, 'fg1_slope_g'] = max(0, -df.loc[i, 'fg1_slope_2'])**0.5
        df.loc[i, 'fg1_curv_left'] = calc_curv_left(fg1_slope, *(b[i] for b in boundaries), outliers[i])
        df.loc[i, 'fg1_curv_right'] = calc_curv_right(fg1_slope, *(b[i] for b in boundaries), outliers[i])

        
        # combinations with slopes
        df.loc[i, 'slope_rel'] = df.loc[i, 'slope_2'] * df.loc[i, 'average_depth']
        df.loc[i, 'fg1_slope_T'] = df.loc[i, 'fg1_slope_2'] * star_info.loc[i, 'Ts'] 
        df.loc[i, 'fg1_slope_rel'] = df.loc[i, 'fg1_slope_2'] * df.loc[i, 'fg1_average_depth']
        df.loc[i, 'fg1_slope_g_rel'] = df.loc[i, 'fg1_slope_g'] * df.loc[i, 'fg1_average_depth']
        
        for q in [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            df.loc[i, f'slope_q_{q}'] = df.loc[i, 'slope_2'] * df.loc[i, f'q_1_{q}']
            df.loc[i, f'slope_q_{q}_2'] = df.loc[i, 'slope_2'] * df.loc[i, f'q_2_{q}']

        
        # other features
        df.loc[i, 't14'] = upper[i] - lower[i]
        df.loc[i, 't23'] = mid2[i] - mid1[i]
        df.loc[i, 'time'] = (mid1[i] - lower[i]) / (upper[i] - lower[i])        
        df.loc[i, 'P_mul_Rs'] = star_info.loc[i, 'P'] * star_info.loc[i, 'Rs']
        df.loc[i, 'P_div_Rs'] = star_info.loc[i, 'P'] / star_info.loc[i, 'Rs']
        
        step = 5
        max_rel = 0
        min_rel = 1
        meaning = 60 # window size for frequency averagning
        
        for j in range(1, signal.shape[-1] - meaning + 1, step):
            if j <= 80:
                meaning = 20
            elif j <= 180:
                meaning = 30
            else:
                meaning = 60

            cur_mean = signal[i, :, j : min(j + meaning, signal.shape[-1])].mean(-1)

            # median filter
            if not outliers[i]:
                if j >= 180:
                    med_kernel = 31
                else:
                    med_kernel = 21
                cur_mean = median_filter(cur_mean, size=med_kernel, mode="constant")
                           
            cur_mean = savgol_filter(cur_mean, 11, 1)
            
            df.loc[i, f'averaged_{j}_unstable'], df.loc[i, f'mid_{j}_unstable'] = calc_depth_and_detrend(cur_mean.copy(), *(b[i] for b in boundaries), 
                                                                                                            outliers[i],
                                                                                                            unstable=True)  
            if not outliers[i]:
                df.loc[i, f'averaged_{j}_unstable'] = 1 - (1 - df.loc[i, f'averaged_{j}_unstable']) * norm_coef
  
            if j in good_waves:
                df.loc[i, f'averaged_{j}'], df.loc[i, f'mid_{j}'] = calc_depth_and_detrend(cur_mean, *(b[i] for b in boundaries),
                                                                                              outliers[i],
                                                                                              fixed_degree=3)
                if not outliers[i]:
                    df.loc[i, f'averaged_{j}'] = 1 - (1 - df.loc[i, f'averaged_{j}']) * norm_coef_2

            
            # percentiles
            for q in [0.1, 0.15, 0.2]:
                if j in good_waves and not outliers[i]:
                    df.loc[i, f'q_w_{j}_{q}'] = calc_perc(cur_mean, mid1[i], mid2[i], q, outliers[i])
                elif outliers[i] and mid1[i] < mid2[i]:
                    x_combined = np.concatenate([np.arange(max(lower[i] - buf1, 1)), np.arange(min(upper[i] + buf1, max_len), max_len + 1)], axis=-1)
                    mid_q = np.quantile(cur_mean[mid1[i] : mid2[i]], q)
                    df.loc[i, f'q_w_{j}_{q}'] = 1 - mid_q / cur_mean[x_combined].mean()
                else:
                    df.loc[i, f'q_w_{j}_{q}'] = 0
           
            
            max_rel = max(max_rel, df.loc[i, f'averaged_{j}_unstable'])
            min_rel = min(min_rel, df.loc[i, f'averaged_{j}_unstable'])

  
            # slope combinations
            df.loc[i, f'averaged_slope_{j}'] = df.loc[i, f'averaged_{j}_unstable'] * df.loc[i, 'slope_2']
            df.loc[i, f'averaged_slope_g_{j}'] = df.loc[i, f'averaged_{j}_unstable'] * df.loc[i, 'slope_g']

        
        # large amplitude     
        if max_rel - min_rel >= 0.005:
            df.loc[i, 'very_bad'] = True

    
    df['Rs'] = star_info['Rs']
    df['Ms'] = star_info['Ms']
    df['Ts'] = star_info['Ts']
    df['sma'] = star_info['sma']  
    df['g'] = np.log10(star_info['Ms'] / (star_info['Rs']**2))
    df['g_T'] = df['g'] * star_info['Ts']
    df['big_rs'] = (star_info['Rs'] > np.quantile(star_info['Rs'].values, 0.97))
    
    df['outliers'] = outliers

    df = df.fillna(0)
    
    return outliers, df