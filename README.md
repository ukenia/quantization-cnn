# quantization-cnn
This repository will host the code for the project 'Quantization of CNN-based Language Models' which is a part of Intro to Deep Learning course at CMU.

To Evaluate with an already trained model. 
```
python evaluate.py --model ./models/wav2letter_31.pth --datadir ./HW4/data/ --batchsize 64
```

The data folder contains the Dataset class and the dataloaders used for training.
The data - train data, train transcripts, validation data and validation labels should be present in the data folder

The `model.py` file contains the Wav2Letter model and the `train.py` file is used for training. To train a model, change the hyperparameters in the `hyperparameters.py` file and run
```
python train.py
```
The `EDADataset.ipynb` file contains the exploratory data analysis on HW4 data.

The quantization folder contains the code used to export the trained model into an ONNX model and quantize the model using ONNX runtime. The inference can be done on the quantized .onnx models. Use the `evaluate.py` with the `--quantized` flag to save the quantized model and run the inference.
Currently, there is only a CPU support for the quantized models.

The utils folder contains all the utils to convert the transcripts into labels and the utils required for training.
