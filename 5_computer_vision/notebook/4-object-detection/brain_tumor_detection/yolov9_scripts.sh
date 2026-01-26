cd yolov9_venv
# uv init
# uv add -r ../yolov9/requirements.txt
. ./.venv/bin/activate

cd ../yolov9

python detect_dual.py \
    --source ../yolov9_test_images/test_image_2.jpg \
    --img 640 \
    --device cpu \
    --weights ../yolov9-c.pt \
    --name yolov9_c_640_detect

python train_dual.py \
    --workers 1 \
    --device cpu \
    --batch 2 \
    --data ../brain_tumor_mri_dataset_coco.yaml \
    --img 640 \
    --cfg models/detect/yolov9-c.yaml \
    --weights '' \
    --name yolov9-c \
    --hyp hyp.scratch-high.yaml \
    --min-items 0 \
    --epochs 50 \
    --close-mosaic 15
