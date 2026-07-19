import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKERBOARD = (9, 6)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []
used_fnames = []

images = sorted(glob.glob("images/left*.jpg"))
print(f"Found {len(images)} images")

gray_shape = None
corners_vis_saved = False
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used_fnames.append(fname)
        print(f"  {fname}: corners found")

        if not corners_vis_saved:
            corners_vis = img.copy()
            cv2.drawChessboardCorners(corners_vis, CHECKERBOARD, corners_refined, found)
            cv2.imwrite("detected_corners_sample.jpg", corners_vis)
            print(f"Saved detected_corners_sample.jpg (from {fname})")
            corners_vis_saved = True
    else:
        print(f"  {fname}: corners NOT found (skipped)")

print(f"\nUsing {len(objpoints)} / {len(images)} images for calibration")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None
)

print("\n=== Calibration Results ===")
print(f"Reprojection error (lower is better, <1.0 is good): {ret:.4f}")
print(f"Camera matrix:\n{camera_matrix}")
print(f"Distortion coefficients:\n{dist_coeffs}")

np.savez("calibration_results.npz",
         camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, reprojection_error=ret)
print("\nSaved results to calibration_results.npz")

sample = cv2.imread(images[0])
undistorted = cv2.undistort(sample, camera_matrix, dist_coeffs)
cv2.imwrite("undistorted_sample.jpg", undistorted)
print("Saved undistorted_sample.jpg for visual comparison")

# Side-by-side original vs. undistorted
side_by_side = np.hstack((sample, undistorted))
cv2.putText(side_by_side, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(side_by_side, "Undistorted", (sample.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite("undistortion_comparison.jpg", side_by_side)
print("Saved undistortion_comparison.jpg (side-by-side)")

# Per-image reprojection error
per_image_error = []
for i in range(len(objpoints)):
    projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    detected = imgpoints[i].reshape(-1, 2)
    projected = projected.reshape(-1, 2)
    error = np.linalg.norm(detected - projected, axis=1).mean()
    per_image_error.append(error)

plt.figure(figsize=(10, 5))
labels = [f.split("/")[-1] for f in used_fnames]
plt.bar(labels, per_image_error, color="steelblue")
plt.axhline(ret, color="red", linestyle="--", label=f"Overall RMS error: {ret:.4f}")
plt.xlabel("Image")
plt.ylabel("Reprojection error")
plt.title("Per-image reprojection error")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("reprojection_error_per_image.png", dpi=150)
print("Saved reprojection_error_per_image.png")
