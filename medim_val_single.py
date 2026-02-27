# -*- encoding: utf-8 -*-

import medim

from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics, print_computed_metrics

if __name__ == "__main__":
    ''' 1. prepare the pre-trained model with local path or huggingface url '''
    # ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    # or you can use a local path like:
    ckpt_path = "./work_dir/train_smile_14subs/sam_model_latest.pth"
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    num_clicks = 1

    ''' 2. read and pre-process your input data '''
    img_path = "./test_data/mra_sub01/image/"
    gt_path = "./test_data/mra_sub01/label/sub01.nii"
    out_path = "./test_data/mra_sub01/sam_out/sub01_pred.nii"
    
    ''' 3. infer with the pre-trained SAM-Med3D model '''
    print("Validation start! plz wait for some times.")
    validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=num_clicks)
    print("Validation finish! plz check your prediction.")

    ''' 4. compute the metrics of your prediction with the ground truth '''
    metrics = compute_metrics(
        gt_path=gt_path,
        pred_path=out_path,
        metrics=['dice'],
        classes=None,
    )
    print_computed_metrics(metrics)
