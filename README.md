# Google Brain - Ventilator Pressure Prediction: Winner's Solution

This repo contains the PID matching code, a fundamental part of the winning solution for the [Google Brain - Ventilator Pressure Prediction](https://www.kaggle.com/c/ventilator-pressure-prediction/) competition. A write-up can be found [here](https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285256).

The code of our two models can be found here:
1) [LSTM on raw data](https://github.com/whoknowsB/google-brain-ventilator-pressure-prediction)
2) [LSTM + CNN Transformer on features](https://github.com/Shujun-He/Google-Brain-Ventilator) which is available on [Kaggle](https://www.kaggle.com/shujun717/1-solution-lstm-cnn-transformer-1-fold) as well.


The matching code is available in `find_pressure_w_triangle.py` but is rather messy. For a better explanation of the concepts, please check out our [notebook on Kaggle](https://www.kaggle.com/group16/1-solution-pid-controller-matching-v1). The `Create Script` notebook can be used to create bash scripts that will do the matching on different CPU's of your computer.
