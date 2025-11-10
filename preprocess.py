import pandas as pd
import numpy as np
import multiprocessing as mp
import itertools
import os
import subprocess
from astropy.stats import sigma_clip
from numpy.polynomial import Polynomial
from tqdm import tqdm
import torch
import torch.nn.functional as F


ROOT = "/kaggle/input/ariel-data-challenge-2025/"
VERSION = "v2"
A_BINNING = 15
F_BINNING = 12*15


device = ("cuda:0" if torch.cuda.is_available() else "cpu")

MODE = os.getenv('PREPROCESS_MODE')


sensor_sizes_dict = {
    "AIRS-CH0": [[11250, 32, 356], [32, 356]],
    "FGS1": [[135000, 32, 32], [32, 32]],
}  # input, mask

cl = 8
cr = 24


def get_gain_offset():
    """
    Get the gain and offset for a given planet and sensor

    Unlike last year's challenge, all planets use the same adc_info.
    We can just hard code it.
    """
    gain = 0.4369
    offset = -1000.0
    return gain, offset



def read_data(planet_id, sensor, mode):
    """
    Read the data for a given planet and sensor
    """
    # get all noise correction frames and signal
    signal = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_signal_0.parquet",
        engine="pyarrow",
    )
    dark_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_0/dark.parquet",
        engine="pyarrow",
    )
    dead_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_0/dead.parquet",
        engine="pyarrow",
    )
    linear_corr_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet",
        engine="pyarrow",
    )
    flat_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_0/flat.parquet",
        engine="pyarrow",
    )

    # reshape to sensor shape and cast to float64
    signal = signal.values.astype(np.float64).reshape(sensor_sizes_dict[sensor][0])[
        :, cl:cr, :
    ]
    dark_frame = dark_frame.values.astype(np.float64).reshape(
        sensor_sizes_dict[sensor][1]
    )[cl:cr, :]
    
    dead_frame = dead_frame.values.reshape(sensor_sizes_dict[sensor][1])[cl:cr, :]
    
    flat_frame = flat_frame.values.astype(np.float64).reshape(
        sensor_sizes_dict[sensor][1]
    )[cl:cr, :]

    linear_corr = linear_corr_frame.values.astype(np.float64).reshape(
        [6] + sensor_sizes_dict[sensor][1]
    )[:, cl:cr, :]

    return (
        signal,
        dark_frame,
        dead_frame,
        linear_corr,
        flat_frame,
    )


def ADC_convert(signal, gain, offset):
    """
    Step 1: Analog-to-Digital Conversion (ADC) correction

    The Analog-to-Digital Conversion (adc) is performed by the detector to convert the
    pixel voltage into an integer number. We revert this operation by using the gain
    and offset for the calibration files 'train_adc_info.csv'.
    
    signal = (signal - offset) * gain
    """

    signal /= gain
    signal += offset
    return signal


def mask_hot_dead(signal, dead, dark):
    """
    Step 2: Mask hot/dead pixel

    The dead pixels map is a map of the pixels that do not respond to light and, thus,
    can't be accounted for any calculation. In all these frames the dead pixels are
    masked using python masked arrays. The bad pixels are thus masked but left
    uncorrected. Some methods can be used to correct bad-pixels but this task,
    if needed, is left to the participants.
    """

    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))

    signal[dead] = np.nan
    signal[hot] = np.nan
    return signal


def apply_linear_corr(c, signal):
    """
    Step 3: linearity Correction

    The non-linearity of the pixels' response can be explained as capacitive leakage
    on the readout electronics of each pixel during the integration time. The number
    of electrons in the well is proportional to the number of photons that hit the
    pixel, with a quantum efficiency coefficient. However, the response of the pixel
    is not linear with the number of electrons in the well. This effect can be
    described by a polynomial function of the number of electrons actually in the well.
    The data is provided with calibration files linear_corr.parquet that are the
    coefficients of the inverse polynomial function and can be used to correct this
    non-linearity effect.
    Using horner's method to evaluate the polynomial
    """
    assert c.shape[0] == 6  # Ensure the polynomial is of degree 5

    return (
        (((c[5] * signal + c[4]) * signal + c[3]) * signal + c[2]) * signal + c[1]
    ) * signal + c[0]


def clean_dark(signal, dark, dt):
    """
    Step 4: dark current subtraction

    The data provided include calibration for dark current estimation, which can be
    used to pre-process the observations. Dark current represents a constant signal
    that accumulates in each pixel during the integration time, independent of the
    incoming light. To obtain the corrected image, the following conventional approach
    is applied: The data provided include calibration files such as dark frames or
    dead pixels' maps. They can be used to pre-process the observations. The dark frame
    is a map of the detector response to a very short exposure time, to correct for the
    dark current of the detector.

    image - (dark * dt)

    The corrected image is conventionally obtained via the following: where the dark
    current map is first corrected for the dead pixel.
    """

    dark = torch.tile(dark, (signal.shape[0], 1, 1))
    signal -= dark * dt[:, None, None]
    return signal


def get_cds(signal):
    """
    Step 5: Get Correlated Double Sampling (CDS)

    The science frames are alternating between the start of the exposure and the end of
    the exposure. The lecture scheme is a ramp with a double sampling, called
    Correlated Double Sampling (CDS), the detector is read twice, once at the start
    of the exposure and once at the end of the exposure. The final CDS is the
    difference (End of exposure) - (Start of exposure).
    """

    return torch.subtract(signal[1::2, :, :], signal[::2, :, :])


def correct_flat_field(flat, signal):
    """
    Step 6: Flat Field Correction

    The flat field is a map of the detector response to uniform illumination, to
    correct for the pixel-to-pixel variations of the detector, for example the
    different quantum efficiencies of each pixel.
    """

    return signal / flat


def bin_obs(signal, binning):
    """
    Step 5.1: Bin Observations

    The data provided are binned in the time dimension. The binning is performed by
    summing the signal over the time dimension.
    """

    cds_binned = torch.zeros(
        (
            signal.shape[0] // binning,
            signal.shape[1],
            signal.shape[2],
        ),
        device=device,
    )
    for i in range(signal.shape[0] // binning):
        cds_binned[i, :, :] = torch.sum(
            signal[i * binning : (i + 1) * binning, :, :], axis=0
        )
    return cds_binned


def nan_interpolation(tensor):
    # Assume tensor is of shape (batch, height, width)
    nan_mask = torch.isnan(tensor)

    # Replace NaNs with zero temporarily
    tensor_filled = torch.where(
        nan_mask, torch.tensor(0.0, device=tensor.device), tensor
    )

    # Create a binary mask (0 where NaNs were and 1 elsewhere)
    ones = torch.ones_like(tensor, device=tensor.device)
    weight = torch.where(nan_mask, torch.tensor(0.0, device=tensor.device), ones)

    # Perform interpolation by convolving with a kernel
    # using bilinear interpolation
    kernel = torch.ones(1, 1, 1, 3, device=tensor.device, dtype=tensor.dtype)

    # Apply padding to the tensor and weight to prevent boundary issues
    tensor_padded = F.pad(
        tensor_filled.unsqueeze(1), (1, 1, 0, 0), mode="replicate"
    ).squeeze(1)
    weight_padded = F.pad(weight.unsqueeze(1), (1, 1, 0, 0), mode="replicate").squeeze(
        1
    )

    # Convolve the filled tensor and the weight mask
    tensor_conv = F.conv2d(tensor_padded.unsqueeze(1), kernel, stride=1)
    weight_conv = F.conv2d(weight_padded.unsqueeze(1), kernel, stride=1)

    # Compute interpolated values (normalized by weights)
    interpolated_tensor = tensor_conv / weight_conv

    # Apply the interpolated values only to the positions of NaNs
    result = torch.where(nan_mask, interpolated_tensor.squeeze(1), tensor)

    return result



def process_planet(planet_id):
    """
    Process a single planet's data
    """
    axis_info = pd.read_parquet(ROOT + "/axis_info.parquet")
    dt_airs = axis_info["AIRS-CH0-integration_time"].dropna().values

    for sensor in ["AIRS-CH0", "FGS1"]:
        # load all data for this planet and sensor
        signal, dark_frame, dead_frame, linear_corr, flat_frame = read_data(
            planet_id, sensor, mode=MODE
        )
        gain, offset = get_gain_offset()

        # Step 1: ADC correction
        signal = ADC_convert(signal, gain, offset)

        # Step 2: Mask hot/dead pixel
        signal = mask_hot_dead(signal, dead_frame, dark_frame)

        # clip at 0
        signal = torch.tensor(signal.clip(0)).to(device)    
        signal = apply_linear_corr(
            torch.tensor(linear_corr).to(device), signal.clone().detach()
        )

        if sensor == "FGS1":
            dt = torch.ones(len(signal), device=device) * 0.1
        elif sensor == "AIRS-CH0":
            dt = torch.tensor(dt_airs).to(device)

        dt[1::2] += 0.1

        signal = clean_dark(signal, torch.tensor(dark_frame).to(device), dt)

        # Step 5: Get Correlated Double Sampling (CDS)
        signal = get_cds(signal)

        # Step 6: Flat Field Correction

        if sensor == "AIRS-CH0":
            signal = bin_obs(signal, binning=A_BINNING)
        else:
            signal = bin_obs(signal, binning=F_BINNING)

        signal = correct_flat_field(torch.tensor(flat_frame).to(device), signal)

        # Step 7: Interpolate NaNs (twice!)
        signal = nan_interpolation(signal)
        signal = nan_interpolation(signal)

        if sensor == "FGS1":
            signal = torch.nanmean(signal, axis=[1, 2]).cpu().numpy()
        elif sensor == "AIRS-CH0":
            signal = torch.nanmean(signal, axis=1).cpu().numpy()
            
        # save the processed signal
        np.save(
            str(planet_id) + "_" + sensor + f"_signal_{VERSION}.npy",
            signal.astype(np.float64),
        )


if __name__ == "__main__":
    star_info = pd.read_csv(ROOT + f"/{MODE}_star_info.csv", index_col="planet_id")
    planet_ids = [int(x) for x in star_info.index.tolist()]

    # Use up to 4 threads!
    mp.set_start_method('spawn')
    with mp.Pool(processes=4) as pool:
        list(tqdm(pool.imap(process_planet, planet_ids), total=len(planet_ids)))

    
    signal_train = []

    for planet_id in planet_ids:
        f_raw = np.load(f"{planet_id}_FGS1_signal_{VERSION}.npy")
        a_raw = np.load(f"{planet_id}_AIRS-CH0_signal_{VERSION}.npy")

        # flip a_raw
        signal = np.concatenate([f_raw[:, None], a_raw[:, ::-1]], axis=1)
        signal_train.append(signal)

        os.remove("/kaggle/working/" + str(planet_id) + f"_AIRS-CH0_signal_{VERSION}.npy")
        os.remove("/kaggle/working/" + str(planet_id) + f"_FGS1_signal_{VERSION}.npy")

    signal_train = np.array(signal_train)
    np.save(f"signal_{VERSION}.npy", signal_train, allow_pickle=False)

    print("Processing complete!")