# BSnet
- Boolean Structured Deep Learning Network
- The network is designed to mimic monotone circuit in Boolean algebra theory 
- Tell the LeNet bully to get lost
- No data augmentations, no regularization such as weight decay and dropout

# Main Takeaways

- My model has 30000 parameters compared to LeNet 60000 parameters
- Model based on the theory of monotone circuit of Boolean algebra
- Under certain conditions, the training optimization function is convex
- Use fully connected layers without overfitting
- Able to be trained on a laptop without GPU
- Able to achieve 80% classification accuracy after masking 70% of the input image pixels to value 1. A normal deep learning model can only achieve 60%

# How to Run

The commands are designed to run on Windows OS. If you are using Linux, adapt the commands accordingly.

Run the command to train a BSnet
```
python keras_first_network_bsnet.py >> bsnet.txt
```

Run the command to train a Normal Relu Network
```
python keras_first_network_normal.py >> normal.txt
```

Run the command to plot out the accuracies curves
```
python plot_acc.py
```

# Model

![Network design](https://github.com/singkuangtan/BSnet/blob/main/bsnet.png)

# Experiment Results 

![Experiment results](https://github.com/singkuangtan/BSnet/blob/main/acc.png)

# Links
[BSnet paper link](https://vixra.org/abs/2212.0193)

[BSautonet paper link](https://vixra.org/abs/2212.0208)

[BSautonet GitHub](https://github.com/singkuangtan/BSautonet)

[Discrete Markov Random Field Relaxation](https://vixra.org/abs/2112.0151)

[Slideshare](https://www.slideshare.net/SingKuangTan)

That's it. 
Have a Nice Day!!!
