# system packages
sudo apt-get install -y libhdf5-serial-dev gmsh python3-tk mpich

# eigen library
wget -c https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz 
mkdir lib
tar -xvf eigen-3.3.9.tar.gz -C lib/
rm eigen-3.3.9.tar.gz

# python packages
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt

# set up the makefile	
cp makefile_linux makefile

# generate the executable
make 
