cd yolov9_venv
# uv init
# uv add -r ../yolov9/requirements.txt
. ./.venv/bin/activate

cd ../yolov9

python detect_dual.py \
    --source ../yolov9_test_images/test_image_2.jpg \
    --img 640 \
    --device cpu \
    --weights ../ckpt_yolov9_pretrained_coco.pt \
    --name ckpt_yolov9_pretrained_coco_detect


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


python detect_dual.py \
    --source ../brain_tumor_mri_dataset/test/y170_jpg.rf.a8208cf093d8dde49b3857e1381c190b.jpg \
    --img 640 \
    --device cpu \
    --weights ../runs_yolov9_brain_tumor_mri_dataset/weights/best.pt \
    --name runs_yolov9_brain_tumor_mri_dataset_detect
