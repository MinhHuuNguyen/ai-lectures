cd Pytorch-UNet_venv
# uv init
# uv add -r ../Pytorch-UNet/requirements.txt
# uv add torch==2.3.0 torchvision==0.18.0 urllib3==1.26.15
. ./.venv/bin/activate
cd ..
python -c "import torch; torch.hub.set_dir('.'); net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)"

cd ../Pytorch-UNet

# 1. Change device in train.py to CPU (line 88)
# From: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# To: device = torch.device('cpu')

python predict.py \
    --model ../checkpoints/unet_carvana_scale0.5_epoch2.pth \
    --input ../carvana_image_masking_dataset/test/0a0e3fb8f782_01.jpg \
    --output ../0a0e3fb8f782_01_output.jpg \
    --scale 0.5

# 1. Change data path in train.py (line 22-23)
# From:
# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# To:
# dir_img = Path('../carvana_image_masking_dataset/train/')
# dir_mask = Path('../carvana_image_masking_dataset/train_masks/')

# 2. Comment out all wandb related code in train.py
# line 16, 58-62, 123-127. 137-140, 146-160

# 3. Change device in train.py to CPU (line 191)
# From: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# To: device = torch.device('cpu')

python train.py --epochs 50 --batch-size 2
