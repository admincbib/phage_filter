#!/bin/bash
python decontaminator/prepare_ds.py configs/test_installation_config.yaml
echo "finished preparation of the training dataset"
python decontaminator/train.py configs/test_installation_config.yaml
echo "finished training the model"
python decontaminator/predict.py configs/test_installation_config.yaml
echo "finished prediction on the test file"
echo "If there were no errors, Decontaminator works properly!"