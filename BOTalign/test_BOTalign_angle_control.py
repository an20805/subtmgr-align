import time

import mrcfile
import numpy as np
from aspire.utils.rotation import Rotation
from aspire.volume import Volume
from numpy.linalg import norm
from numpy.random import normal

from utils_BO import align_BO, get_angle, generate_data, u_to_rot



# Test volume and noise level.
# Examples:
# data_name = "0.1/0.mrc"
data_name = "fixed.mrc"
# data_name = "emd-3683.mrc"
inv_SNR = 0


# BOTalign parameters:
# [loss type, downsampling level, number of iterations, refine]
para = ["wemd", 91, 150, False]


USE_RANDOM_ROTATION = False
fixed_angle_deg = 10.0


def random_unit_axis():
    axis = normal(0, 1, 3)
    axis_norm = norm(axis)
    while axis_norm == 0:
        axis = normal(0, 1, 3)
        axis_norm = norm(axis)
    return axis / axis_norm


def generate_data_with_fixed_angle(data_name, inv_SNR, angle_deg):
    with mrcfile.open("data/" + data_name) as mrc:
        template = Volume(mrc.data)

    L = template.shape[1]
    shape = (L, L, L)
    ns_std = np.sqrt(inv_SNR * norm(template) ** 2 / L**3)

    vol0 = template + np.float32(normal(0, ns_std, shape))

    axis = random_unit_axis()
    angle_rad = np.deg2rad(angle_deg)
    R_true = np.float32(u_to_rot(axis * angle_rad))

    vol_given = template.rotate(Rotation(R_true)) + np.float32(normal(0, ns_std, shape))

    return vol0, vol_given, L, R_true, axis


if USE_RANDOM_ROTATION:
    vol0, vol_given, L, R_true = generate_data(data_name, inv_SNR)
    true_angle_deg = get_angle(R_true, np.eye(3))
    axis = None
else:
    vol0, vol_given, L, R_true, axis = generate_data_with_fixed_angle(
        data_name, inv_SNR, fixed_angle_deg
    )
    true_angle_deg = fixed_angle_deg


tic = time.perf_counter()
R_init, R_rec = align_BO(vol0, vol_given, para)
toc = time.perf_counter()


angle_gap_init = get_angle(R_init, R_true.T)
angle_gap_rec = get_angle(R_rec, R_true.T)


print("BOTalign single-run test")
print(f"Volume: {data_name}")
print(f"Volume size: {L}")
print(f"Inverse SNR: {inv_SNR}")
print(f"Parameters: {para}")
print(f"Random rotation mode: {USE_RANDOM_ROTATION}")
print(f"True rotation magnitude (deg): {true_angle_deg:.4f}")
if axis is not None:
    print(f"Rotation axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
print(f"Runtime (s): {toc - tic:.4f}")
print(f"Initial estimate angle gap (deg): {angle_gap_init:.6f}")
print(f"Recovered rotation angle gap (deg): {angle_gap_rec:.6f}")
