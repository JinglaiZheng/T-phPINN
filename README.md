# T-phPINN:First order time derivative enhanced parallel hard constraints physics-informed neural networks
The code for the paper Improved physics-informed neural networks for solving 2D non-Fourier heat conduction equation
# Code
The code for solving non-fourier heat conduction equation by hPINN based on deepxde in main.py
# Abstract
Non-Fourier heat conduction plays a dominant role in many extreme transient heat conduction processes. In order to accurately solve the non-Fourier heat conduction equation in 2D domain, a first order time derivative enhanced parallel hard-constraint physics-informed neural networks (T-phPINN) is proposed. T-phPINN containing two sub-networks brings in the first order time derivative to capture the rate of temperature change, and a special first order time derivative approximation term is added to the total loss function. The results of two numerical cases show that the relative error of T-phPINN is 1.04% and 12.30% of traditional PINN respectively, which demonstrates the superiority of our architecture. A transfer learning framework is established for T-phPINN, and the training under different equation parameters only requires 1/6 iterations of the basic model, and close accuracy is obtained. The results show great potential of the framework for solving 2D non-Fourier heat conduction equations expeditiously and precisely.
# Citation
If you use this data or code for academic research, you are encouraged to cite the following paper:
@article{
  title={Improved physics-informed neural networks for solving 2D non-Fourier heat conduction equation},
  author={JinglaiZheng,FanLi,and Haiming Huang},
  journal={International Journal of Heat and Mass Transfer},
  volume={#},
  pages={#},
  year={2024},
  publisher={Elsevier}
  }
