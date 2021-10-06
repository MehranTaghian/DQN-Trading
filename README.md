# DQN-Trading

This is a framework based on deep reinforcement learning for stock market trading. This project is the implementation
code for the two papers:

- [Learning financial asset-specific trading rules via deep reinforcement learning](https://arxiv.org/abs/2010.14194)
- [A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules](https://arxiv.org/abs/2101.03867)

The deep reinforcement learning algorithm used here is Deep Q-Learning.

## Acknowledgement

- [Deep Q-Learning tutorial in pytorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## Requirements

Install pytorch using the following commands. This is for CUDA 11.1 and python 3.8:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- python = 3.8
- pandas = 1.3.2
- numpy = 1.21.2
- matplotlib = 3.4.3
- cython = 0.29.24
- scikit-learn = 0.24.2

## TODO List

- [X] Right now this project does not have a code for getting user hyper-parameters from terminal and running the code.
  We preferred writing a jupyter notebook (`Main.ipynb`) in which you can set the input data, the model, along with
  setting the hyper-parameters.

- [X] The project also does not have a code to do Hyper-parameter search (its easy to implement).

- [X] You can also set the seed for running the experiments in the original code for training the models.

## Developers' Guidelines

In this section, I briefly explain different parts of the project and how to change each. The data for the project
downloaded from [Yahoo Finance](http://finance.yahoo.com/) where you can search for a specific market there and download
your data under the `Historical Data` section. Then you create a directory with the name of the stock under the data
directory and put the `.csv`
file there.

The `DataLoader` directory contains files to process the data and interact with the RL agent. The `DataLoader.py` loads
the data given the folder name under `Data` folder and also the name of the `.csv` file. For this, you should use
the `YahooFinanceDataLoader` class for using data downloaded from Yahoo Finance.

The `Data.py` file is the environment that interacts with the RL agent. This file contains all the functionalities used
in a standard RL environment. For each agent, I developed a class inherited from the Data class that only differs in the
state space (consider that states for LSTM and convolutional models are time-series, while for other models are simple
OHLCs). In `DataForPatternBasedAgent.py` the states are patterns extracted using rule-based methods in technical
analysis. In `DataAutoPatternExtractionAgent.py`
states are Open, High, Low, and Close prices (plus some other information about the candle-stick like trend, upper
shadow, lower shadow, etc). In `DataSequential.py` as it is obvious from the name, the state space is time-series which
is used in both LSTM and Convolutional models. `DataSequencePrediction.py` was an idea for feeding states that have been
predicted using an LSTM model to the RL agent. This idea is raw and needs to be developed.

Where ever we used encoder-decoder architecture, the decoder is the DQN agent whose neural network is the same across
all the models.

The `DeepRLAgent` directory contains the DQN model without encoder part (`VanillaInput`) whose data loader corresponds
to `DataAutoPatternExtractionAgent.py` and `DataForPatternBasedAgent.py`; an encoder-decoder model where the encoder is
a 1d convolutional layer added to the decoder which is DQN agent under `SimpleCNNEncoder` directory; an encoder-decoder
model where encoder is a simple MLP model and the decoder is DQN agent under `MLPEncoder` directory.

Under the `EncoderDecoderAgent` there exist all the time-series models, including `CNN`
(two-layered 1d CNN as encoder), `CNN2D` (a single-layered 2d CNN as encoder), `CNN-GRU`
(the encoder is a 1d `CNN` over input and then a `GRU` on the output of `CNN`. The purpose of this model is that `CNN`
extracts features from each candlestick, then`GRU`
extracts temporal dependency among those extracted features.), `CNNAttn` (A two-layered 1d CNN with attention layer for
putting higher emphasis on specific parts of the features extracted from the time-series data), and a `GRU` encoder
which extracts temporal relations among candles. All of these models use `DataSequential.py` file as environment.

For running each agent, please refer to the `Main.ipynb` file for instructions on how to run each agent and feed data.
The `Main.ipynb` file also has code for plotting results.

The `Objects` directory contains the saved models from our experiments for each agent.

The `PatternDetectionCandleStick` directory contains `Evaluation.py` file which has all the evaluation metrics used in
the paper. This file receives the actions from the agents and evaluate the result of the strategy offered by each agent.
The `LabelPatterns.py` uses rule-based methods to generate buy or sell signals. Also `Extract.py`
is another file used for detecting wellknown candlestick patterns.

`RLAgent` directory is the implementation of the traditional RL algorithm SARSA-&#955; using cython. In order to run
that in the `Main.ipynb` you should first build the cython file. In order to do that, run the following script inside
it's directory in terminal:

```bash
python setup.py build_ext --inplace
```

This works for both linux and windows.

For more information on the algorithms and models, please refer to the original paper. You can find them in the
references.

If you had any questions regarding the paper, code, or you wanted to contribute,
please send me an email: taghianmehran@gmail.com


## References

```
@article{taghian2020learning,
  title={Learning financial asset-specific trading rules via deep reinforcement learning},
  author={Taghian, Mehran and Asadi, Ahmad and Safabakhsh, Reza},
  journal={arXiv preprint arXiv:2010.14194},
  year={2020}
}

@article{taghian2021reinforcement,
  title={A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules},
  author={Taghian, Mehran and Asadi, Ahmad and Safabakhsh, Reza},
  journal={arXiv preprint arXiv:2101.03867},
  year={2021}
}
```