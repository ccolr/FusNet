python run_all.py \
    --weights     fusnet_new_7_outputs/last_model.pth \
    --test_txt    test.txt \
    --data_dir    . \
    --image_dir   bamboo/images \
    --gt_dir      bamboo/labels \
    --output_root visual_results_res_swin_mamba