cd yolov9
uv init
uv add -r requirements.txt
. ./.venv/bin/activate

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
    --epochs 5 \
    --close-mosaic 15
