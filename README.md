# learn_for_crowd2

### Guide

#### In

```python
import tensorflow as tf

from lc import *
from tensorflow.contrib.layers import *
from tensorflow.contrib.keras.python.keras.layers import *

config.LEARNING_RATE = 0.001   
config.DECAY_STEP = 50  
config.DECAY_RATE = 0.96
config.L2_LAMBDA = 5
config.STOP_THRESHOLD = -1

l = Loader("test_2")
```

```python
def five_layers_lrelu(x,ref_y, test):
    test = None if not test else True
    lrelu = LeakyReLU()
    hid1 = fully_connected(x, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer1")
    hid2 = fully_connected(hid1, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer2")
    hid3 = fully_connected(hid2, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer3")
    hid4 = fully_connected(hid3, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer4")
    hid5 = fully_connected(hid4, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer5")
    y = fully_connected(hid5, 1, activation_fn=tf.identity, reuse=test, scope="fc")
    if not test:
        analysis.add_RMSE_loss(y, ref_y, "train")
        #analysis.add_L2_loss()
    else:
        analysis.add_RMSE_loss(y, ref_y, "test")

def linear(x,ref_y, test):
    test = None if not test else True
    y = fully_connected(x, 1, activation_fn=tf.identity, reuse=test, scope="fc")
    if not test:
        analysis.add_RMSE_loss(y, ref_y, "train")
        #analysis.add_L2_loss()
    else:
        analysis.add_RMSE_loss(y, ref_y, "test")

def apply_graph(graph):
    g1 = tf.Graph()
    with g1.as_default():
        x1, y1 = l.train()
        graph(x1, y1, False)

        x2, y2 = l.validation()
        graph(x2, y2, True)

        summarize_collection("trainable_variables")
        summarize_collection("losses")    
    return g1
```

```python
with apply_graph(five_layers_lrelu).as_default():
    train.simple_train(500000)
```

#### Out
```
Epoch loop 1: loss 78.3254, train accuracy 78.3254, cross validation accuracy 78.4047
Epoch loop 2: loss 84.0987, train accuracy 84.0987, cross validation accuracy 84.0185
Epoch loop 3: loss 102.7358, train accuracy 102.7358, cross validation accuracy 102.8177
Epoch loop 4: loss 70.9299, train accuracy 70.9299, cross validation accuracy 71.0026
Epoch loop 5: loss 23.7462, train accuracy 23.7462, cross validation accuracy 23.7850
Epoch loop 6: loss 91.9496, train accuracy 91.9496, cross validation accuracy 91.9396
Epoch loop 7: loss 36.8563, train accuracy 36.8563, cross validation accuracy 36.8253
Epoch loop 8: loss 44.7049, train accuracy 44.7049, cross validation accuracy 44.7571
Epoch loop 9: loss 56.9053, train accuracy 56.9053, cross validation accuracy 56.9668
Epoch loop 10: loss 48.5368, train accuracy 48.5368, cross validation accuracy 48.5988
Epoch loop 11: loss 29.7577, train accuracy 29.7577, cross validation accuracy 29.8145
Epoch loop 12: loss 9.7990, train accuracy 9.7990, cross validation accuracy 9.7249
Epoch loop 13: loss 15.2850, train accuracy 15.2850, cross validation accuracy 15.2186
Epoch loop 14: loss 9.9996, train accuracy 9.9996, cross validation accuracy 10.0241
Epoch loop 15: loss 10.4510, train accuracy 10.4510, cross validation accuracy 10.4749
Epoch loop 16: loss 10.1662, train accuracy 10.1662, cross validation accuracy 10.1073
Epoch loop 17: loss 6.5611, train accuracy 6.5611, cross validation accuracy 6.5071
Epoch loop 18: loss 13.6231, train accuracy 13.6231, cross validation accuracy 13.6477
Epoch loop 19: loss 10.9389, train accuracy 10.9389, cross validation accuracy 10.9557
Epoch loop 20: loss 11.2738, train accuracy 11.2738, cross validation accuracy 11.2300
Epoch loop 21: loss 9.5614, train accuracy 9.5614, cross validation accuracy 9.5150
Epoch loop 22: loss 11.4296, train accuracy 11.4296, cross validation accuracy 11.4487
Epoch loop 23: loss 12.2073, train accuracy 12.2073, cross validation accuracy 12.2292
Epoch loop 24: loss 5.8522, train accuracy 5.8522, cross validation accuracy 5.7974
Epoch loop 25: loss 9.5259, train accuracy 9.5259, cross validation accuracy 9.4659

... ...

Epoch loop 837: loss 2.5107, train accuracy 2.5107, cross validation accuracy 2.5709
Epoch loop 838: loss 2.5039, train accuracy 2.5039, cross validation accuracy 2.5919
Epoch loop 839: loss 2.5068, train accuracy 2.5068, cross validation accuracy 2.5684
Epoch loop 840: loss 2.5011, train accuracy 2.5011, cross validation accuracy 2.5910
Epoch loop 841: loss 2.5064, train accuracy 2.5064, cross validation accuracy 2.5673
Epoch loop 842: loss 2.5031, train accuracy 2.5031, cross validation accuracy 2.5915
Epoch loop 843: loss 2.5094, train accuracy 2.5094, cross validation accuracy 2.5713
Epoch loop 844: loss 2.5054, train accuracy 2.5054, cross validation accuracy 2.5959
Epoch loop 845: loss 2.5106, train accuracy 2.5106, cross validation accuracy 2.5715
Epoch loop 846: loss 2.5045, train accuracy 2.5045, cross validation accuracy 2.5934
Epoch loop 847: loss 2.5074, train accuracy 2.5074, cross validation accuracy 2.5699
Epoch loop 848: loss 2.5009, train accuracy 2.5009, cross validation accuracy 2.5918
```