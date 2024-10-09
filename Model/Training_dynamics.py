train_res_recon_error = []
train_res_perplexity = []
betas = []
ent = []

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
    
    
    #beta = 32
    lambd = 0.001#*beta   

     #/  
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
        
    vq_loss, data_recon, perplexity,entropy,encodings = model(data,beta,lambd)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())
    ent.append(entropy.item())
    betas.append(beta)
    
    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print('beta',beta)



from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# Smooth the values
train_res_recon_error_smooth = uniform_filter1d(train_res_recon_error, size=10, mode='nearest')
train_res_perplexity_smooth = uniform_filter1d(train_res_perplexity, size=10, mode='nearest')
betas_smooth = uniform_filter1d(betas, size=10, mode='nearest')
entropy_smooth = uniform_filter1d(ent, size=10, mode='nearest')

# Create a figure with 3 subplots (1 row, 3 columns)
f, ax = plt.subplots(1, 3, figsize=(16, 4))

# Plot on the left y-axis (first subplot)
ax[0].plot(train_res_recon_error_smooth, label='Smoothed NMSE')
ax[0].set_yscale('log')
ax[0].set_title('Smoothed NMSE')

# Create a twin axis for the first subplot for the beta values
ax0_twin = ax[0].twinx()
ax0_twin.plot(betas_smooth, color='red', label='Beta')
ax0_twin.set_ylabel('Beta')

# Plot the perplexity on the second subplot
ax[1].plot(train_res_perplexity_smooth, label='Smoothed Average codebook usage (perplexity)')
ax[1].set_title('Smoothed Average codebook usage (perplexity)')

# Create a twin axis for the second subplot for the beta values
ax1_twin = ax[1].twinx()
ax1_twin.plot(betas_smooth, color='red', label='Beta')
ax1_twin.set_ylabel('Beta')

# Plot the entropy on the third subplot
ax[2].plot(entropy_smooth, label='Smoothed Entropy')
ax[2].set_title('Smoothed Entropy')

ax1_twin = ax[2].twinx()
ax1_twin.plot(betas_smooth, color='red', label='Beta')
ax1_twin.set_ylabel('Beta')
# Adjust layout and show the plot
plt.tight_layout()
plt.show()