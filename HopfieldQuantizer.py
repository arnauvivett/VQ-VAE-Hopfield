class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs,beta,lambd):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1,self._embedding_dim)
        
        #flat input : dim (BHW,embedding_dim)
        #embedding.weight.t : dim (embedding_dim,num_embeddings)
        similarity = torch.matmul(flat_input, self._embedding.weight.t()) #dim (BHW,num_embeddings) 
        
        #* Encoding softmax Enc = softmax(embedding encoding)encoding 
        #beta = 1/np.sqrt(self._num_embeddings) 
        #print(beta)
        
        encodings = F.softmax(beta*similarity, dim=1) 
        
        ##############
        encodings_max = torch.argmax(encodings, dim=1)
        
        encodings_max = torch.multinomial(encodings, 1).squeeze(1)  # Sampling from the softmax distribution

        encodings_hot = torch.zeros_like(encodings).scatter_(1, encodings_max.unsqueeze(1), 1)

        encodings = encodings + (encodings_hot - encodings).detach()
        ##############
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) 

        avg_probs = torch.mean(encodings, dim=0)
        
        dkl = (avg_probs * torch.log(avg_probs + 1e-10)) 

        entropy = -torch.sum(encodings * torch.log(encodings + 1e-15), dim=-1) #worst
        entropy2 = beta*torch.sum(encodings * similarity, dim=-1)
        dkl1 =     beta*torch.sum(encodings*similarity, dim=0) 

        #* Loss 
        loss = F.mse_loss(quantized, inputs) + lambd*torch.mean(entropy2)
        
        #e_latent_loss = F.mse_loss(quantized.detach(), inputs) 
        #q_latent_loss = F.mse_loss(quantized, inputs.detach()) 
        #loss = q_latent_loss + self._commitment_cost * e_latent_loss 
        
        avg_probs = torch.mean(encodings, dim=0)

        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) 
        #quantized = inputs + quantized
        quantized = inputs + (quantized - inputs).detach() 
        # convert quantized from BHWC -> BCHW 
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, lambd*torch.mean(entropy2)
