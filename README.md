# Autonomous-Leveling

## Overview
A simulation platform for autonomous soil leveling.


## Clone Repo

```bash
git clone https://github.com/yourusername/Autonomous-Leveling.git
cd Autonomous-Leveling
```
remember to do `git lfs update` to get the NN weights.
## Usage

### Install Chrono Simulation Engine
Noticed you need to download chrono engine with following link https://drive.google.com/file/d/1EuYlXzaYqnN3c9kwzDyVE9AnakoO-Es1/view?usp=sharing instead of official website page of chrono because it has the latest and faster terramechanics development.

After downloading the zip file, upzip it and try to install the simulation engine

```bash
cd chrono-engine && mkdir build && cd build
```

Run ccmake or cmake gui to build and install the chrono engine. you need to enable these following modules enabled: Vehicle, FSI, VSG. The installation process would be similar as described in official Chrono document: https://api.projectchrono.org/tutorial_install_chrono.html. 

### Build the chrono demo for autonomous leveling
Once the chrono simulation engine is installed, we can link the built chrono engine and build autonomous leveling demo: `sim/soil_leveling.cpp` 

```bash
cd sim && mkdir build && cd build && ccmake ../ -G Ninja
```
Make sure to set chrono path correctly: `Chrono_DIR = <path_to_chrono>/chrono-engine/build/cmake`. Then build the code by running:
```bash
ninja
```
After successfully building demo, there will be an executable called: `soil_leveling_sim` in your folder `sim/build/`

### Conda install to run NN based algorithm

create and activate conda environment

```bash
conda create -n autograding python=3.10 && conda activate autograding
```
install the pytorch related package through conda
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
install other supporting packages
```bash
conda install numpy matplotlib
```
### Runing simulation with our autonomous blade control algorithm:

1. make sure you are in the `autograding` conda environment. Runing algorithm:
```bash
python algo/auto_level_sim.py --pile_height 0.37 
```
2. after blade control algorithm is spinning, start the chrono simulation from another terminal:
```bash
cd sim/ && ./build/soil_leveling_sim --pile_height 0.37
```
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.