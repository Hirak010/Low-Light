import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import logging
import argparse
from tqdm import tqdm # Optional: for progress bar

# --- Configuration ---
# Setup argument parser
parser = argparse.ArgumentParser(description='Calculate PSNR, SSIM, LPIPS between images in two folders.')
parser.add_argument('--gt_dir', type=str, default='output5', help='Directory containing ground truth images.')
parser.add_argument('--output_dir', type=str, default='output_combined', help='Directory containing model output images.')
parser.add_argument('--log_file', type=str, default='New.log', help='File to save the metrics log.')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU for LPIPS calculation if available.')
args = parser.parse_args()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(args.log_file),
        logging.StreamHandler() # Also print logs to console
    ]
)
logger = logging.getLogger(__name__)

# --- LPIPS Model Setup ---
# Check for GPU availability if requested
use_gpu = args.use_gpu and torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
logger.info(f"Using device: {device} for LPIPS calculation.")

# Load LPIPS model (choose 'alex' or 'vgg')
# If using GPU, move the model to GPU
try:
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
except Exception as e:
    logger.error(f"Error initializing LPIPS model: {e}")
    logger.error("Please ensure the 'lpips' library and its dependencies (torch, torchvision) are installed correctly.")
    exit(1)


# --- Helper Functions ---
def preprocess_image_for_lpips(img_path, device):
    """Loads and preprocesses an image for LPIPS calculation."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            return None
        # Convert BGR (OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert HWC to CHW format
        img = img.transpose((2, 0, 1))
        # Convert to float tensor, normalize to [0, 1], then to [-1, 1]
        img_tensor = torch.from_numpy(img).float().to(device) / 255.0
        img_tensor = (img_tensor * 2.0) - 1.0
        # Add batch dimension (N, C, H, W)
        return img_tensor.unsqueeze(0)
    except Exception as e:
        logger.error(f"Error processing image {img_path} for LPIPS: {e}")
        return None

# --- Main Calculation Logic ---
def calculate_metrics():
    gt_folder = args.gt_dir
    output_folder = args.output_dir

    if not os.path.isdir(gt_folder):
        logger.error(f"Ground truth directory not found: {gt_folder}")
        return
    if not os.path.isdir(output_folder):
        logger.error(f"Output directory not found: {output_folder}")
        return

    gt_files = sorted([f for f in os.listdir(gt_folder) if os.path.isfile(os.path.join(gt_folder, f))])
    output_files = sorted([f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))])

    # Ensure we only process files present in both folders with the same name
    common_files = sorted(list(set(gt_files) & set(output_files)))

    if not common_files:
        logger.warning("No common image files found in both directories.")
        return

    logger.info(f"Found {len(common_files)} matching images to process.")

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    # Use tqdm for progress bar if installed
    file_iterator = tqdm(common_files, desc="Calculating Metrics") if 'tqdm' in globals() else common_files

    for filename in file_iterator:
        gt_path = os.path.join(gt_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # --- Load images for PSNR/SSIM (using OpenCV) ---
            img_gt_cv = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            img_out_cv = cv2.imread(output_path, cv2.IMREAD_COLOR)

            if img_gt_cv is None:
                logger.warning(f"Skipping {filename}: Could not read GT image {gt_path}")
                continue
            if img_out_cv is None:
                logger.warning(f"Skipping {filename}: Could not read Output image {output_path}")
                continue

            # --- Basic Checks ---
            if img_gt_cv.shape != img_out_cv.shape:
                logger.warning(f"Skipping {filename}: Image dimensions mismatch. "
                               f"GT: {img_gt_cv.shape}, Output: {img_out_cv.shape}")
                continue

            # --- Calculate PSNR ---
            # data_range is max value - min value (typically 255 for uint8 images)
            current_psnr = psnr(img_gt_cv, img_out_cv, data_range=255)
            psnr_scores.append(current_psnr)

            # --- Calculate SSIM ---
            # For color images, set multichannel=True (older scikit-image)
            # or channel_axis=-1 (newer scikit-image >= 0.19)
            # Using channel_axis is preferred. data_range=255 for uint8
            # win_size must be odd and <= min(height, width), common values are 7 or 11
            win_size = min(7, img_gt_cv.shape[0], img_gt_cv.shape[1])
            if win_size % 2 == 0: # Ensure win_size is odd
                win_size -= 1
            if win_size < 3:
                logger.warning(f"Skipping SSIM for {filename}: Image too small for default window size.")
                current_ssim = np.nan # Indicate skip
            else:
                 current_ssim = ssim(img_gt_cv, img_out_cv, data_range=255, channel_axis=-1, win_size=win_size)
            if not np.isnan(current_ssim):
                 ssim_scores.append(current_ssim)


            # --- Calculate LPIPS ---
            img_gt_lpips = preprocess_image_for_lpips(gt_path, device)
            img_out_lpips = preprocess_image_for_lpips(output_path, device)

            if img_gt_lpips is None or img_out_lpips is None:
                logger.warning(f"Skipping LPIPS for {filename} due to preprocessing errors.")
                current_lpips = np.nan # Indicate skip
            else:
                # Ensure tensors are on the correct device (redundant if preprocess handles it)
                img_gt_lpips = img_gt_lpips.to(device)
                img_out_lpips = img_out_lpips.to(device)

                with torch.no_grad(): # Important: disable gradient calculation
                    current_lpips = loss_fn_alex(img_gt_lpips, img_out_lpips).item()
                lpips_scores.append(current_lpips)

            # Log individual results
            logger.info(f"{filename}: PSNR={current_psnr:.4f}, SSIM={current_ssim:.4f}, LPIPS={current_lpips:.4f}")

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            # Optionally continue to next file or stop execution
            continue # Continue processing other files

    # --- Calculate and Log Average Metrics ---
    if psnr_scores:
        avg_psnr = np.mean(psnr_scores)
        logger.info(f"\n--- Average Metrics ({len(psnr_scores)} images) ---")
        logger.info(f"Average PSNR: {avg_psnr:.4f}")
    else:
        logger.info("\nNo PSNR scores calculated.")

    valid_ssim_scores = [s for s in ssim_scores if not np.isnan(s)]
    if valid_ssim_scores:
        avg_ssim = np.mean(valid_ssim_scores)
        logger.info(f"Average SSIM: {avg_ssim:.4f} ({len(valid_ssim_scores)} images)")
    else:
        logger.info("No valid SSIM scores calculated.")

    valid_lpips_scores = [s for s in lpips_scores if not np.isnan(s)]
    if valid_lpips_scores:
        avg_lpips = np.mean(valid_lpips_scores)
        logger.info(f"Average LPIPS: {avg_lpips:.4f} ({len(valid_lpips_scores)} images)")
    else:
        logger.info("No valid LPIPS scores calculated.")

    logger.info(f"Metrics log saved to: {args.log_file}")

# --- Run the Calculation ---
if __name__ == "__main__":
    calculate_metrics()