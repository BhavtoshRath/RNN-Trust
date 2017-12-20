This project contains Recurrent Neural Network implementations (basic, GRU and LSTM) using Theano library.
It models embeddings of users in a twitter network in sequential manner. For example, retweet network,
based on the sequence in which users retweeted a source tweet.

Below is the model:
![RNN-Trust](https://github.com/BhavtoshRath/RNN-Trust/blob/master/RNN_model.png)

In train-theano.py file,
Initialize training and testing datasets in lines 9-12.
Choose any of the three models in lines 68-70.

More details about the model can be found in my ASONAM, 17 paper. Please cite if it was helpful in your research. Thanks.

@inproceedings{Rath2017FromRT,
  title={From Retweet to Believability: Utilizing Trust to Identify Rumor Spreaders on Twitter},
  author={Bhavtosh Rath and Wei Gao and Jing Ma and Jaideep Srivastava},
  booktitle={ASONAM},
  year={2017}
}