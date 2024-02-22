import torch
import random
x=torch.zeros(3,48,64)
rand=[ random.randint(0,48)  for i in range(30) ]
y=x[:,rand,:]


print(y.shape)




