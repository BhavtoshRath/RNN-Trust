import time
from datetime import datetime
from utils import *
from BasicRNN import RNNTheano
from LSTM_RNN import LSTMTheano
from GRU_RNN import GRUTheano
from evaluate import *

X_train = ''
y_train = ''
X_test = ''
y_test = ''


_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '200'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '100'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')


def train_with_sgd(model, X_train, y_train, X_test, y_test, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them late r
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_total_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()

            # Test model
            prediction = []
            for j in X_test:
                temp = model.predict(j)

                temp1 = []
                for i in temp:
                    temp1.append(i.argmax())
                prediction.append(temp1)

                # prediction.append(temp)
            res = evaluation(prediction, y_test)
            print res
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.emb_dim, time),
                                         model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


# model = LSTMTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
# model = GRUTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
# model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)

t1 = time.time()

t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, y_train, X_test, y_test, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)