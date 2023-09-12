# Step 1:
lsblk

# Step 2: Create partition
sudo fdisk /dev/sda

# Step 3: In the interactive shell, type the following commands
# n => create a new partition.
# p => primary partition
# others are default values
# w => write changes to disk and exit

# Step 3: Format partition
sudo mkfs.ext4 /dev/sda1

# Step 4: Mount partition
sudo mkdir /mnt/disk
sudo mount /dev/sda1 /mnt/disk

# Step 5: Check if mounted
df -h

# Step 6: Unmount partition
sudo umount /mnt/disk
