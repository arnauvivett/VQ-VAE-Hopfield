                 
mnist_transform = transforms.Compose([
        transforms.ToTensor(),])

kwargs = {'num_workers': 1, 'pin_memory': True} 

training_data = MNIST(dataset_path, train=True, download=True,
                     transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])) 
                         
validation_data  = MNIST(dataset_path, train=False, download=True, 
                     transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])) 
                         
data_variance = np.var(training_data.data.numpy() / 255.0) 


training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=True)

