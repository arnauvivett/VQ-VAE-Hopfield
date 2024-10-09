train_res_recon_error = []
train_res_perplexity = []
betas = []
entropy = []

model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)


num_training_updates = 4000

model.train()

for i in tqdm(xrange(num_training_updates)):
    (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()
    
    
    beta = 64
    lambd = 0.0001#*beta     
    beta = 32* ((i+1)/num_training_updates)
    
    """   #\/
    if i < num_training_updates/2:
        beta = 9.1-2*9*(i+1)/num_training_updates 
    elif i > num_training_updates/2:
        beta = 0.1+2*9*(i-num_training_updates/2)/num_training_updates 
    """
    
    """  #/\
    if i < num_training_updates/2:
        beta = 1+16*(i+1)/num_training_updates 
    elif i > num_training_updates/2:
        beta = 9-16*(i-num_training_updates/2)/num_training_updates 
    """
    """  #_/
    if i < num_training_updates/2:
        beta = 2.5
    elif i > num_training_updates/2:
        beta = 2.5+16*(i-num_training_updates/2)/num_training_updates 
    """
        
    vq_loss, data_recon, perplexity,encodings = model(data,beta,lambd)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())
    entropy.append(encodings.item())
    betas.append(beta)
    
    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print('beta',beta)

