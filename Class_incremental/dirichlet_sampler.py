import numpy as np


np.random.seed(0)

def sampler(train_data, n_clients = 5, n_classes = [] , alpha = 0.1, use_balance = True):
    n_samples_train = len(train_data)

    ##### Determine locations of different classes
    all_ids_train = np.array(train_data.targets)
    class_ids_train = {class_num: np.where(all_ids_train == class_num)[0] for class_num in n_classes}

    ##### Determine distribution over classes to be assigned per client
    # Returns n_clients x n_classes matrix
    dist_of_client = np.random.dirichlet(np.repeat(alpha, n_clients), size=len(n_classes)).transpose()
    dist_of_client /= dist_of_client.sum()

    if(use_balance):
        for i in range(100):
            s0 = dist_of_client.sum(axis=0, keepdims=True)
            s1 = dist_of_client.sum(axis=1, keepdims=True)
            dist_of_client /= s0
            dist_of_client /= s1

    ##### Allocate number of samples per class to each client based on distribution
    samples_per_class_train = (np.floor(dist_of_client * n_samples_train))
    
    start_ids_train = np.zeros((n_clients+1,len(n_classes)), dtype=np.int32)
    for i in range(0, n_clients):
        start_ids_train[i+1] = start_ids_train[i] + samples_per_class_train[i]




    ##### Save IDs
    # Train
    client_ids = {client_num: {} for client_num in range(n_clients)}
    for client_num in range(n_clients):
        l = np.array([], dtype=np.int32)
        for i,class_num in enumerate(n_classes):
            start, end = start_ids_train[client_num, i], start_ids_train[client_num+1, i]
            l = np.concatenate((l, class_ids_train[class_num][start:end].tolist())).astype(np.int32)
        client_ids[client_num] = l


    return client_ids
