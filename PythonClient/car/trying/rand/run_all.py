import subprocess
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
IGNORE_LABEL = 255
NUM_CLASSES = 50  # Updated to 50 as per your comment
LOG_DIR = 'log'   # Directory where your checkpoints are saved

def calculate_iou(confusion_matrix):
    """
    Calculates the IoU per class from a confusion matrix.
    Returns the IoU array and the ground truth sums.
    """
    tp = np.diag(confusion_matrix)
    gt_sum = confusion_matrix.sum(axis=1)
    pred_sum = confusion_matrix.sum(axis=0)
    union = gt_sum + pred_sum - tp
    
    # Avoid division by zero, output NaN for missing classes
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / union
        
    return iou, gt_sum

def print_iou_results(iou_array, gt_sum, title="Results"):
    """
    Helper function to print the IoU array.
    """
    print(f"\n--- {title} ---")
    print(f"{'Class ID':<10} {'IoU (%)':<10} {'Count (Points)':<15}")
    print("-" * 35)
    
    valid_iou_list = []
    
    for i in range(NUM_CLASSES):
        if gt_sum[i] > 0 and not np.isnan(iou_array[i]):
            score = iou_array[i] * 100
            print(f"{i:<10} {score:<10.2f} {gt_sum[i]:<15}")
            valid_iou_list.append(score)
            
    if valid_iou_list:
        mIoU = np.mean(valid_iou_list)
        print("-" * 35)
        print(f"Mean IoU (mIoU): {mIoU:.2f}%")
        print("=" * 35)
    else:
        print("No valid classes found to evaluate.")

def evaluate_sequence(test_id):
    """
    Calculates confusion matrix for a specific sequence ID.
    Returns: confusion_matrix (np.array)
    """
    gt_dir = os.path.join('output', str(test_id), 'labels')
    pred_dir = os.path.join('result', str(test_id), 'predictions')
    
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Skipping Eval for {test_id}: Directories not found.")
        return confusion_matrix

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])
    
    for file_name in tqdm(gt_files, desc=f"Seq {test_id} Eval", leave=False):
        gt_path = os.path.join(gt_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)
        
        if not os.path.exists(pred_path):
            continue

        gt = np.load(gt_path).reshape(-1)
        pred = np.load(pred_path).reshape(-1)

        if gt.shape != pred.shape:
            continue

        valid_mask = (gt != IGNORE_LABEL) & (gt < NUM_CLASSES) & (pred < NUM_CLASSES)
                     
        gt = gt[valid_mask]
        pred = pred[valid_mask]

        flat_inds = gt * NUM_CLASSES + pred
        count = np.bincount(flat_inds, minlength=NUM_CLASSES**2)
        confusion_matrix += count.reshape(NUM_CLASSES, NUM_CLASSES)
    
    return confusion_matrix

def plot_results(mean_ious, std_ious):
    """
    Generates a bar chart with error bars.
    """
    print("\nGenerating evaluation plot...")
    plt.figure(figsize=(16, 6))
    
    classes = np.arange(1, NUM_CLASSES + 1)
    
    # Create bar chart with error bars
    plt.bar(
        classes, mean_ious, 
        yerr=std_ious, 
        capsize=4, 
        color='skyblue', 
        edgecolor='black',
        error_kw={'elinewidth': 1.5, 'alpha': 0.8}
    )
    
    plt.xlabel('Class ID')
    plt.ylabel('IoU (%)')
    plt.title('Average IoU per Class across 5 Runs')
    plt.xticks(classes, rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_img = 'average_iou_error_plot.png'
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved successfully to {output_img}")
    plt.show()

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    test_id_all = [0, 1, 2, 3, 4, 5, 6, 7]
    all_runs_ious = []

    for j in range(1, 6):
        print(f"\n{'#'*50}")
        print(f" EVALUATING MODEL RUN {j} ")
        print(f"{'#'*50}")
        
        checkpoint_file = os.path.join(LOG_DIR, f"checkpoint_slope_final_{j}.tar")
        run_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        
        for test_id in test_id_all:
            print(f"Processing Sequence {test_id} for Model {j}...")
            
            # 1. Run Inference 
            # Note: You MUST pass the checkpoint_path so test_SemanticKITTI loads the right model!
            try:
                subprocess.run(
                    [
                        'python', 'test_SemanticKITTI.py', 
                        '--test_id', str(test_id),
                        '--checkpoint_path', checkpoint_file
                    ],
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f"Test with test_id={test_id} failed. Skipping.")
                continue
            
            # 2. Add sequence matrix to this run's global matrix
            seq_cm = evaluate_sequence(test_id)
            run_confusion_matrix += seq_cm

        # 3. Calculate and store IoU for Model J
        iou, gt_sum = calculate_iou(run_confusion_matrix)
        print_iou_results(iou, gt_sum, title=f"Run {j} Global Results")
        
        # Multiply by 100 to store as percentage
        all_runs_ious.append(iou * 100)

    # ==========================================
    # FINAL STATISTICAL CALCULATION
    # ==========================================
    # Convert to 2D numpy array (shape: 5 runs x NUM_CLASSES)
    iou_matrix = np.array(all_runs_ious)
    
    # Calculate mean and std dev, ignoring NaNs (classes that never appeared)
    mean_ious = np.nanmean(iou_matrix, axis=0)
    std_ious = np.nanstd(iou_matrix, axis=0)
    # save the mean_ious and std_ious to a file
    np.save('mean_ious.npy', mean_ious)
    np.save('std_ious.npy', std_ious)
    print("\n\n" + "#" * 50)
    print(" FINAL AVERAGE RESULTS ACROSS ALL 5 RUNS ")
    print("#" * 50)
    for i in range(NUM_CLASSES):
        if not np.isnan(mean_ious[i]):
            print(f"Class {i:<4} | Mean IoU: {mean_ious[i]:<6.2f}% ± {std_ious[i]:<5.2f}%")
            
    # Generate the plot with standard deviation error bars
    plot_results(mean_ious, std_ious)