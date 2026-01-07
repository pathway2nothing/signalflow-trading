After PR to main:

git checkout main
git pull

git tag -a vx.x.x -m "Release vx.x.x"
git push origin vx.x.x

sudo apt-get install -y \
  libgtk-3-0 libdrm2 libx11-6 libxext6 libxrender1 \
  fonts-liberation fontconfig
