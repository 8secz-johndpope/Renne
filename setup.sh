apt update
apt upgrade
apt install -y libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev
apt install -y python3-pip
git clone https://github.com/Joinn99/Renne
pip3 install -r requirements.txt 
cd Renne
cd models
wget https://www.dropbox.com/s/nah0kwsralwda0b/pose2seg-full.pth
wget https://www.dropbox.com/s/9ubmrc0chgedgza/openpose-pytorch.pth
cd ..