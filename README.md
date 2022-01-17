# Numerical experiments for application of Markov Chain Monte Carlo to establishment of calibration interval  
Contributor : Hyung-Seok Shim and Seung-Nam Park (They are with Korean Research Institute for Standards and Science)  

This repository contains jupyter notebook (**main.ipynb**) and modules(**\*.py**) to implement NCSLI S3 method and Markov Chain Monte Carlo(MCMC) method for obtaining calibration interval. These files were used to derive the results presented in the paper **"Numerical experiments for application of Markov Chain Monte Carlo to establishment of calibration intervals" (DOI : 10.1109/TIM.2022.3142005)**.

**main.ipynb** : main code to implement numerical experiments  
**rel_mode.py** : modules containing functions for reliability models  
**data_gen.py** : modules containing functions for generating hypothetical calibration records  
**parameter_estimation.py** : modules containing functions for estimating parameters of reliability model using S3 and MCMC method  
**distribution.py** : modules containing functions for presenting distribution of parematers and reliability model  
**calibration_interval.py** : modules containing functions for calculating optimal calibration interval, and presenting distribution of optimal calibration interval  
**correlation.py** : modules containing functions for presenting joint distribution of parameters, and calculating correlation coefficient  

Any comments related to this notebook are welcome. (e-mail : shimhs@kriss.re.kr)
