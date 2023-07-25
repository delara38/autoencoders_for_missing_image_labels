import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.distributions import Categorical
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ImageEncoder(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3,6,5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6,16,5)
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84, 10)


  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x


class ImageDecoder(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(10,84)
    self.fc2 = nn.Linear(84,120)
    self.fc3 = nn.Linear(120, 16*5*5)
    self.fc4 = nn.Linear(16*5*5, 32*32*3)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x


class ClassifierNet(ImageEncoder):
  def __init__(self):
    super().__init__()
    
  def forward(self, x):
    x = super().forward(x)
    x = F.softmax(x)

    return x
  
def train_autoencoder(epochs=5):

  net = ImageEncoder()
  
  decoder = ImageDecoder()

  
  net_optimizer = optim.Adam(list(net.parameters())+ list(decoder.parameters()), lr=0.001)
  for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader):

      inputs, labels = data

      net_optimizer.zero_grad()


      representation = net(inputs)
      image_reconstruction = decoder(representation)

      target = torch.flatten(inputs, 1)

      loss = F.mse_loss(image_reconstruction, target)
      loss.backward()

      net_optimizer.step()

      running_loss += loss.item()

      if i % 2000 == 1999:
        print(f'[{epoch + 1}, {i + 1:5d}], loss:[{running_loss/2000:.3f}')
        running_loss=0

  return net, decoder

def train_classifier(epochs=5):

   classifier = ClassifierNet()

   criterion = nn.CrossEntropyLoss()
   net_optimizer = optim.Adam(classifier.parameters(), lr=0.002)

   for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader):

      inputs, labels = data

      net_optimizer.zero_grad()


      
      probs = classifier(inputs)

      loss = F.cross_entropy(probs, labels)
      loss.backward()

      net_optimizer.step()

      running_loss += loss.item()

      if i % 2000 == 1999:
        print(f'[{epoch + 1}, {i + 1:5d}], loss:[{running_loss/2000:.3f}')
        running_loss=0
   return classifier

if __name__ == '__main__':


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )
    batch_size=8
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)



    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                        download=True, transform=transform)


    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    p = list(trainset)

    imgs = torch.stack([imgs[0] for imgs in p])
    lbls = [img[1] for img in p]


    nml = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    #k = nml(imgs).pow(2).sum(axis=[1,2,3])
    #sns.displot(x=k,kind='kde', hue=lbls)


    alphas = [1000,5000, 10000, 14000]
    p_miss = 0.7
    
    for alpha in alphas:
        print(alpha)

        missing = []
        obs = []
        for img in p:

            image, label = nml(img[0]), img[1]


            if image.pow(2).sum() > alpha and np.random.random() < p_miss:
                #then we set it to be missing,

                missing.append((image, label)) 
            else:
                obs.append((image, label))


        obs_imgs = [o[0] for o in obs]
        obs_lbls = np.array([o[1] for o in obs])

        trainloader = torch.utils.data.DataLoader(obs, batch_size=batch_size,
                                                shuffle=False)


        cc = train_classifier(5)

        na, decoder = train_autoencoder(5)

        pc = cc(torch.stack(obs_imgs))
        pa = na(torch.stack(obs_imgs))



        def pmm(image, net, ps,lbls,cut=5):
            """
            image is some image dataset
            """

            x_locs = net(image.unsqueeze(0))[0]

            idks = torch.exp(-1*(torch.linalg.vector_norm(x_locs-ps,dim=1)))
            top100_inds = torch.topk(idks, cut)[1]
            p = F.softmax(idks[top100_inds])

            img_label = np.random.choice(lbls[top100_inds],p=p.detach().numpy() )


            return img_label


        labels = []

        for img in missing:

            image, pred_label_pc = img[0], img[1]

            pred_label_pa = pmm(image, na,pa, obs_lbls)
            pred_label_pc = pmm(image, cc,pc, obs_lbls)
            classifier_probs = cc(image.unsqueeze(0))
            classifier_label = Categorical(classifier_probs).sample().item()


            labels.append((pred_label_pa, pred_label_pc, classifier_label))


        torch.save(na, f"net_autoencoder_alpha_{alpha}.pt")
        torch.save(cc, f'net_classifier_alpha_{alpha}.pt')


        import pickle

        pickle.dump({'missing':missing,'obs':obs, 'pa':pa, 'pc':pc, 'labels':labels}, open(f'pmm_data_alpha_{alpha}.pkl','wb'))