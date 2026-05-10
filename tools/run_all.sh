python tools/run_all.py \
    --model fusnet_legacy_1 \
    --weights fusnet_legacy1_outputs/best_model.pth \
    --test_txt test.txt --data_dir . \
    --image_dir bamboo/images --gt_dir bamboo/labels \
    --output_root results/legacy1
