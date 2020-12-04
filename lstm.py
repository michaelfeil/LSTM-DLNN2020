"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""

#Reference Assignment: https://github.com/quanpn90/LSTMAssignment-DLNN2020

"""
#import cupy did not work for gtx960m and cuda version 10
try:
    
    cupy_running= False
    raise #"not using cupy"

    #import cupy      #for things that can only be done with cupy
    #import cupy as np
    
    #print("CUPY is working. using cupy for np calculations")
    #cupy_running= True
    
except:
    import numpy as np
    print("using numpy for np calculations")
"""


import numpy as np
  
import numpy #for things that can only be done with numpy
import time
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def dtanh(x):
    return 1 - x * x

def relu(x):
   return np.maximum(0,x)

def drelu(y):
    return np.where(y <= 0, 0, 1)


# The numerically stable softmax implementation
def softmax(x):
    # assuming x shape is [feature_size, batch_size]
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


# data I/O
with open('data/input.txt', 'r') as f:
    data = f.read()  # should be simple plain text file


#[Start FUN]
def fun(data, 
        fun_level): #1==boring #1000 = moderate #10000 = high # 100000 = extreme_fun
    """
    inputs: 
    - data:  string, optionaly one of a Shakespeare play
    - fun_level: int, number of passages to add to play
    
    returns 
    - data:  string,  string of a Shakespeare play with new speakers and more fun texts
    
    
    """
    print("start the fun")
    items = ["custom message", "cat", "dog", "universe", "primary school", "neural network", "political joke", "us dollar"]
    name = ["Michael the Great", #add some groeßenwahn too
            "Nuns choir",       
            "Last Citizen", 
            "Dealer"] #append your name!
    
    for i in range(fun_level):  
        
        while True: #while no newline found
            lenght_data = len(data)
                        
            random_index = int(numpy.random.rand()*lenght_data) #random enough
                        
            found_index =  data[random_index:].find("\n\n")  #finding next newline index from random index in data
            if found_index!=-1:
                newline_index = random_index+found_index #return index of a randomized newline 
                break 
            
        message_options = 4
        message_choice = numpy.random.randint(message_options, size=1)[0] #select message format
        
        name_choice = name[numpy.random.randint(len(name), size=1)[0]]
        
        item_choice = items[numpy.random.randint(len(items), size=1)[0]]
        
        #that is just for fun, lets see if it works
        if message_choice == 0:
            special_data="\n\n{}:\nI want this {} to be remembered!\n\n".format(name_choice, item_choice)
        if message_choice == 1:
            special_data="\n\n{}:\nThis {} is all mine,\nso recognize it, please?\n\n".format(name_choice, item_choice)
        if message_choice == 2:
            special_data ="\n\n{}:\nRemembering any {} will be our future.\n\n".format(name_choice, item_choice)
        if message_choice == 3:
            special_data ="\n\n{}:\nWhere is my {}? I want it now!\n\n".format(name_choice, item_choice)
        
        data = data[:newline_index] + special_data + data[2+newline_index:] #random injection of a new speaker/special_data in place where only /n/n was before
        
        if i % 1000 == 0:
            print("adding fun... fun progress: {} / {}".format(i, fun_level))
    
    print("\n here is some fun. Sample training data: \n \t", data[:1000].replace("\n", "\n\t"), "\n")
    return data

have_fun = False

if have_fun: 
    data = fun(data, fun_level= 3000) #add some fun to data

    #data = data.lower() #want to make it easier and convert it all to lowercase?
#[End FUN]

chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

print("dict for chars", ix_to_char)

if len(sys.argv) > 1:
    print(len(sys.argv))
    option = sys.argv[1]
else:
    option = "train"

# hyperparameters
emb_size = 16
hidden_size = 256   # size of hidden layer of neurons
seq_length = 128  # number of steps to unroll the RNN for
learning_rate = 0.05
batch_size = 24
max_epochs = 20
std = 0.1

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size) * std  # embedding layer
bex = np.random.randn(emb_size, 1) * std  # embedding bias

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std  # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std  # input gate
Wo = np.random.randn(hidden_size, concat_size) * std  # output gate
Wc = np.random.randn(hidden_size, concat_size) * std  # c term

bf = np.random.randn(hidden_size, 1) * std  # forget bias
bi = np.random.randn(hidden_size, 1) * std # input bias
bo = np.random.randn(hidden_size, 1) * std # output bias
bc = np.random.randn(hidden_size, 1) * std # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size) * std  # hidden to output
by  = np.random.randn(vocab_size, 1) * std # output bias



# this will load the data into memory
data_stream = np.asarray([char_to_ix[char] for char in data]) # zeitlicher array der indexe für die Charakter
print(data_stream.shape) # vector shape, 1D vector 

bound = (data_stream.shape[0] // (seq_length * batch_size)) * (seq_length * batch_size) # data_stream.shape[0] beschreibt menge aller characters die hintereinander vorkommen
cut_stream = data_stream[:bound] # schneidet das ende ab, so dass alles ins batches passt. 
cut_stream = np.reshape(cut_stream, (batch_size, -1)) # jede spalte ist ein batch mit batch_size elementen. elemente sind characters. (-1) heisst zweite dimension passt sich automatisch an dass alle eintraege aufgetielt sind
print(cut_stream.shape)

# Stop when processed this much data
max_updates_per_epoch = cut_stream.shape[1] - (seq_length *2 ) - 1



def forward(inputs, targets, memory, batch_size=batch_size):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    hprev, cprev = memory
    xs, wes, zhs, ys, ps, hs, cs, h_f, h_i, h_c, h_o, ls = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    
    #initial states if hs and cs
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    

    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, batch_size)) # encode in 1-of-k representation
        ls[t] = np.zeros((vocab_size, batch_size))
        for b in range(batch_size):
            #input for batches
            xs[t][inputs[t][b]][b] = 1 
            #label for batches
            ls[t][targets[t][b]][b] = 1

        # convert word indices to word embeddings
        
        #wes[t] = np.tanh(np.dot(Wex,  xs[t]) + bex) 
        #wes[t] = relu(np.dot(Wex,  xs[t]) + bex) 
        wes[t] = np.dot(Wex,  xs[t]) + bex

        # LSTM cell operation
        # first concatenate the input and h to get z
        zhs[t] = np.vstack((wes[t], hs[t-1]))

        # compute the forget gate
        # f = sigmoid(Wf * zhs + bf)
        h_f[t] = sigmoid(np.dot(Wf, zhs[t]) + bf)

        # compute the input gate
        # i = sigmoid(Wi * z + bi)
        h_i[t] = sigmoid(np.dot(Wi, zhs[t]) + bi)
        
        # compute the candidate memory
        # c_ = tanh(Wc * zhs + bc)
        h_c[t] = np.tanh(np.dot(Wc, zhs[t]) + bc)
        
        # output gate
        #o = sigmoid(Wo * zhs + bo)
        h_o[t] = sigmoid(np.dot(Wo, zhs[t]) + bo)
        
        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_t = f * c_(t-1) + i * c_
        #new cell
        cs[t] = h_f[t] * cs[t-1] + h_i[t] * h_c[t]
        #new hidden
        hs[t] = np.tanh(cs[t]) * h_o[t]
        

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars
        # softmax for probabilities for next chars
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = softmax(ys[t])


        # cross-entropy loss
        loss_t = np.sum(-np.log(ps[t]) * ls[t])
        loss += loss_t
        # loss += -np.log(ps[t][targets[t],0])

    activations = xs, wes, zhs, ys, ps, hs, cs, h_f, h_i, h_c, h_o, ls
    #memory = (hs[-1], cs[-1]) #bug
    input_length = inputs.shape[0]
    memory = (hs[input_length - 1], cs[input_length -1])

    return loss, activations, memory


def backward(activations, clipping=True):
    xs, wes, zhs, ys, ps, hs, cs, h_f, h_i, h_c, h_o, ls = activations

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dbex = np.zeros_like(bex)
    dby = np.zeros_like(by)
    
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bc), np.zeros_like(bo)

    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    

    # back propagation through time starts here
    for t in reversed(range(seq_length)):
        #loss and softmax / labels
        dy = ps[t] - ls[t] #apparently loss of softmax
        
        # Gradient Fully connected output to output. #no activations
        dWhy += np.dot(dy, hs[t].T)                #divide though batch_size for avg
        dby += np.sum(dy, axis=-1, keepdims=True)
        
        
        # through the fully-connected layer (Why, by) to h. 
        #add dhnext from future timestep // backprop through time
        
        dh = np.dot(Why.T, dy) + dhnext

        
        #  gradient of multiplication with output gate: at the output of tanh.
        dctanh = h_o[t] * dh
        
        # through the tanh function; since cs[t] branches in two
        #add dcnext from future timestep // backprop through time
        
        dc = dctanh * dtanh(np.tanh(cs[t])) + dcnext

        # gradient of multiplication with tanh; 
        dhh_o = dh * np.tanh(cs[t]) #gradient at the output of the sigmoid of the output gate
        dho = dhh_o * dsigmoid(h_o[t]) # back through the sigmoid  

        # Compute gradients for the output gate 
        dWo += np.dot(dho, zhs[t].T)
        dbo += np.sum(dho, axis=-1, keepdims=True)
        dzh_dho = np.dot(Wo.T, dho)

        # Compute gradients for the forget gate 
        dhf = cs[t-1] * dc * dsigmoid(h_f[t]) 
        dWf += np.dot(dhf, zhs[t].T)
        dbf += np.sum(dhf, axis=-1, keepdims=True)
        dzh_dhf = np.dot(Wf.T, dhf)

        # Compute gradients for the input gate 
        dhi = h_c[t] * dc * dsigmoid(h_i[t])
        dWi += np.dot(dhi, zhs[t].T)
        dbi += np.sum(dhi, axis=-1, keepdims=True)
        dzh_dhi = np.dot(Wi.T, dhi)
        
        # Compute gradients for the candidate memory gate 
        dhc = h_i[t] * dc * dtanh(h_c[t]) 
        dWc += np.dot(dhc, zhs[t].T)
        dbc += np.sum(dhc, axis=-1, keepdims=True)
        dzh_dhc = np.dot(Wc.T, dhc)

        #for dh all need to be combined
        dzh = dzh_dho + dzh_dhf + dzh_dhi + dzh_dhc
        dhnext = dzh[emb_size:, :] # section to hidden state towards the past / dhnext.
        
        # dcnext from dc and the forget gate.
        dcnext = h_f[t] * dc
        #END LSTM #finished backprop through timestep
        
        #Part of gradient towards input at timestep
        dx = dzh[:emb_size, :] # section NOT to hidden state towards the past / dhnext but to input / embedding
        
        
        #finally back trough embedding
        
        #dx = dx*dtanh(wes[t]) #times derivative of tanh activation
        #dx = dx*drelu(wes[t])
        
        
        dWex += np.dot(dx, xs[t].T)
        dbex += np.sum(dx, axis=-1, keepdims=True)

    
    
    for dparam in [dWex, dbex, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWhy, dby]:
        # clip to mitigate exploding gradients
        dparam = dparam/batch_size
        if clipping:
            np.clip(dparam, -5, 5, out=dparam)

    gradients = (dWex, dbex, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWhy, dby)

    return gradients


def sample(seed_ix, n, temperature=0.9):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    
    seed_ix: integer from ix_to_char
    n: number of characters to be sampled
    temperature: float element (0, inf): adjusts elements of original propability distribution: p = p^temperature
        - value near inf: picks always argmax(prediction) p = p^înf
        - value near 0: makes predicted prop distribution are similar 
        - value 1: uses
    returns: generated_chars: an array of indexes
    
    Example: seed_ix = 46, n=4
    returns [5, 10, 10, 1] #hallo
    
    when ix_to_char: {46: "H", 5: "a", 10="l",  15"o"}
    
    """
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        # Run the forward pass 
        
        wes = np.dot(Wex, x)
        zh = np.vstack((wes, h))
        hf = sigmoid(np.dot(Wf, zh) + bf)
        hi = sigmoid(np.dot(Wi, zh) + bi)
        ho = sigmoid(np.dot(Wo, zh) + bo)
        hc = np.tanh(np.dot(Wc, zh) + bc)
        c = hf * c + hi * hc
        h = np.tanh(c) * ho
        y = np.dot(Why, h) + by
        p = softmax(y)
        #end forward pass
    
        # Sample from the distribution produced by softmax.
        exp_preds = np.exp(np.log(p) / temperature)
        adj_propability_distribution = exp_preds / np.sum(exp_preds)
        
        if cupy_running == True:
            adj_propability_distribution = cupy.asnumpy(adj_propability_distribution) #if running with cuda cupy, need to convert it first
        
        ix = numpy.random.choice(range(vocab_size), p=adj_propability_distribution.ravel())
        
               
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(int(ix))
    
    return ixes

if option == 'train':

    epoch, n, p, n_updates, n_updates_prev, not_improved_loss_since = 1, 0, 0, 0, 0, 0
    
    start_time = time.time()
    intermediate_time = time.time()
    
    next_shift = np.arange(seq_length) # shift "slightly" increases randomned of new inputs after an epoch
    np.random.shuffle(next_shift) #randomized starting offsets between 0 and seq_lengh
    
    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mbex, mby = np.zeros_like(bex), np.zeros_like(by)

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
    
    smooth_loss_previous = smooth_loss
    
    data_length = cut_stream.shape[1]

    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= data_length or n == 0:
            hprev = np.zeros((hidden_size, batch_size))  # reset RNN memory
            cprev = np.zeros((hidden_size, batch_size))
            p = 0  # go from start of data

        inputs  =  cut_stream[:, p   : p + seq_length  ].T 
        targets =  cut_stream[:, p+1 : p + seq_length+1].T
        
        assert inputs.shape[0] == seq_length
        assert targets.shape[0] == seq_length
        
        if n % 100 == 0:
            # sample from the model every now and then
            sample_ix = sample(0, 150)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt,))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        hprev, cprev = memory
        gradients = backward(activations)

        dWex, dbex, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss/batch_size * 0.001
        if n % 20 == 0:
            characters_per_second = (n_updates-n_updates_prev)*batch_size*seq_length / (time.time() - intermediate_time)
            n_updates_prev = n_updates #reset counter
            print('iter %d, total sequences %d, \t characters_per_second %d epoch %d, loss: %f' % (n, n_updates*batch_size, characters_per_second, epoch, smooth_loss))  # print progress
            intermediate_time = time.time()
        
        #callbacks
        '''
        if n_updates % 50 == 0:
            
            loss_progress = (smooth_loss_previous - smooth_loss) / smooth_loss_previous
            
            improved_loss = bool(loss_progress > 0.001) #if did improve by more than 0.1%
            
            if improved_loss: 
                #set new goals
                smooth_loss_previous = smooth_loss
                not_improved_loss_since = n_updates 
                
            elif not improved_loss:
                # reduce learning rate
                if n_updates-not_improved_loss_since > 200:
                    lr_decrease = 0.8
                    print("reducing learning rate from {} to {}".format(learning_rate, lr_decrease*learning_rate))
                    learning_rate *= lr_decrease #reducing learning rate
                    
                elif n_updates-not_improved_loss_since > 2000:
                    print("early stopping")
                    break
        '''
            
        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wex,  bex,  Wf,  bf,  Wi,  bi,  Wc,  bc,  Wo,  bo,  Why,  by],
                                      [dWex, dbex, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWhy, dby],
                                      [mWex, mbex, mWf, mbf, mWi, mbi, mWc, mbc, mWo, mbo, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

        p += seq_length  # move data pointer
        n += 1  # iteration counter
        n_updates += 1
        
        if p > max_updates_per_epoch:
            #end of epoch
            epoch += 1
            next_shift = np.roll(next_shift, 1) #change starting offset of stream for the next epoch
            p = 0 + next_shift[0] #starting index of , first epoch at 0
            
            print("for next epoch {},train again with shift by {} ".format(epoch, next_shift[0]))
            if epoch > max_epochs:
                break #break while #end of training
    
    #finished with training, sample some data so everybody is convinced that the LSTM works
    for tmp in [0.2, 0.5, 0.7, 0.9, 1]:   
        sample_ix = sample(0, 2000, temperature=tmp)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('\n \n --Temperature {} Final Sample--\n {} \n--Final Sample--'.format(tmp, txt,))
    
    with open("./results_"+str(int(time.time()))+".txt", "w") as f:
        
        result_samping = " \n Hyperparamters: \n \t batch_size: {} \n \t emb_size: {} \n \t hidden_size: {} \n \t seq_length: {} \n \t learning_rate: {} \n \t epochs: {} \n final loss {} \n ".format(
                                batch_size, emb_size,      hidden_size,            seq_length,                 learning_rate,          max_epochs, smooth_loss)
        f.write(result_samping)
        
        for tmp in [0.2, 0.5, 0.7, 0.9, 1]:   
            sample_ix = sample(0, 2000, temperature=tmp)
            sampling = ''.join(ix_to_char[ix] for ix in sample_ix),
            
            f.write('\n Start Sample temp: {} --- \n'.format(tmp))
            f.write(' %s ' % sampling)
            f.write('\n --- End Sample temp: {} \n \n '.format(tmp))

elif option == 'gradcheck':

    data_length = cut_stream.shape[1]

    p = 0
    # inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    # targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
    inputs = cut_stream[:, p:p + seq_length].T
    targets = cut_stream[:, p + 1:p + 1 + seq_length].T

    delta = 0.0001

    hprev = np.zeros((hidden_size, batch_size))
    cprev = np.zeros((hidden_size, batch_size))

    memory = (hprev, cprev)

    loss, activations, hprev = forward(inputs, targets, memory)
    gradients = backward(activations, clipping=False)
    dWex, dbex, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWhy, dby = gradients

    for weight, grad, name in zip([Wex,  bex,  Wf,  bf,  Wi,  bi,  Wc,  bc,  Wo,  bo,  Why,  by],
                                  [dWex, dbex, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWhy, dby],
                                  ["Wex",  "bex",  "Wf",  "bf",  "Wi",  "bi",  "Wc",  "bc",  "Wo",  "bo",  "Why",  "by"]):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print("checking",name)
        countidx = 0
        gradnumsum = 0
        gradanasum = 0
        relerrorsum = 0
        erroridx = []

        for i in range(weight.size):

            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)
            gradnumsum += grad_numerical
            gradanasum += grad_analytic
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            if rel_error is None:
                rel_error = 0.
            relerrorsum += rel_error

            if rel_error > 0.001:
                print ('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                countidx += 1
                erroridx.append(i)
                
        print('For %s found %i bad gradients; with %i total parameters in the vector/matrix!' % (
            name, countidx, weight.size))
        print(' Average numerical grad: %0.9f \n Average analytical grad: %0.9f \n Average relative grad: %0.9f' % (
            gradnumsum / float(weight.size), gradanasum / float(weight.size), relerrorsum / float(weight.size)))
        print(' Indizes at which analytical gradient does not match numerical:', erroridx)
