apt update
apt upgrade
apt install -y libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev
apt install -y python3-pip
git clone https://github.com/Joinn99/Renne
pip3 install -r requirements.txt 
cd models
wget https://www.dropbox.com/s/nah0kwsralwda0b/pose2seg-full.pth
wget https://www.dropbox.com/s/9ubmrc0chgedgza/openpose-pytorch.pth
wget https://docs.google.com/uc?id=1G8lXquU3eREfs8KorFpFC8N4YmTQRksF&export=download
mv 1G8lXquU3eREfs8KorFpFC8N4YmTQRksF EdgeModel_gen.pth
wget https://docs.google.com/uc?id=1_oYnmK7kppXqka9UUsHrZB4gWE4ouSgT&export=download
mv 1_oYnmK7kppXqka9UUsHrZB4gWE4ouSgT InpaintingModel_gen.pth
cd ..