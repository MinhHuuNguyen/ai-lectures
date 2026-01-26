cd Pytorch-UNet_venv
# uv init
# uv add -r ../Pytorch-UNet/requirements.txt
# uv add torch==2.3.0 torchvision==0.18.0 urllib3==1.26.15
. ./.venv/bin/activate

cd ../Pytorch-UNet
python predict.py -i image.jpg -o output.jpg

# Change data path in train.py
# From:
# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# To:
# dir_img = Path('../carvana_image_masking_dataset/train/')
# dir_mask = Path('../carvana_image_masking_dataset/train_masks/')

# Comment out all wandb related code in train.py

# Change device in train.py to CPU

python train.py --amp --epochs 50 --batch-size 8 --scale 1
