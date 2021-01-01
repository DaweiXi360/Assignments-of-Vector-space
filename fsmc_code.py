import numpy as np


# General function to compute expected hitting time for Exercise 1
def compute_Phi_ET(P, ns=100):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        ns {int} -- largest step to consider

    Returns:
        Phi_list {numpy.array} -- (ns + 1) x n x n, the Phi matrix for time 0, 1, ...,ns
        ET {numpy.array} -- n x n, expected hitting time approximated by ns steps ns
    '''
    Mt = np.eye(P.shape[0],P.shape[1],dtype=float)
    Phi_list = []
    Phi_list.append(Mt)
    ET = np.zeros((P.shape[0],P.shape[1]),dtype=float)
    
    for i in range(ns):
        Mt = np.dot(P,Mt)
        for j in range(P.shape[0]):
            Mt[j][j] = 1.0
        Phi_list.append(Mt)
        ET += (Mt-Phi_list[i])*(i+1)
    
    Phi_list = np.array(Phi_list)
    
    return Phi_list, ET


# General function to simulate hitting time for Exercise 1
def simulate_hitting_time(P, states, nr):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        states {list[int]} -- the list [start state, end state], index starts from 0
        nr {int} -- largest step to consider

    Returns:
        T {list[int]} -- a size nr list contains the hitting time of all realizations
    '''
    Time = []
    choice = []
    for i in range(P.shape[1]):
        choice.append(i)
    
    for i in range(nr):
        init = np.eye(1,P.shape[1],k = 0,dtype=float)
        flag = 0
        count = 0
        for j in range(nr):
            state_dis = np.dot(init,P)
            state_now = np.random.choice(choice,1,replace=True,p=state_dis[0])
            count += 1
            
            if state_now[0] == states[1]:
                Time.append(count)
                flag = 1
                break
            init = np.eye(1,P.shape[1],k=state_now[0],dtype=float)
            
        if flag==0:
            Time.append(0)
    
    return T



# General function to compute the stationary distribution (if unique) of a Markov chain for Exercise 3
def stationary_distribution(P):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain

    Returns:
        pi {numpy.array} -- length n, stationary distribution of the Markov chain
    '''
    I = np.eye(P.shape[0],P.shape[1],dtype=float)
    Ones = np.ones((1,P.shape[1]),dtype=float)
    Temp = np.append(np.transpose(P)-I,ones,axis=0)
    
    b = np.zeros(P.shape[1])
    b = np.append(b,[1],axis=0)
    b = np.transpose(b)
    
    pi = np.linalg.solve(np.dot(np.transpose(Temp),Temp),np.dot(np.transpose(Temp),b))

    return pi
