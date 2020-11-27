import numpy as np
import ROOT
from math import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
from keras import *

class Interval(object):
    def __init__(self, middle, deviation):
        self.lower = middle - abs(deviation)
        self.upper = middle + abs(deviation)

    def __contains__(self, item):
        return self.lower <= item <= self.upper

def getNumpyArrayFromFile(filename, typ):
    #This loads the data we want to train on
    f    = ROOT.TFile.Open(filename)
    tree = f.Get("sf/t") 
    #This is the vector of inputs
    v = []
    tags = []
    #This is the vector of tags
    i = 0
    nEntries = tree.GetEntries()
    for ev in tree:
        i += 1

        LepW=ROOT.TLorentzVector()
        LepW.SetPtEtaPhiM(getattr(ev, "LepW_pt"),getattr(ev, "LepW_eta"),getattr(ev, "LepW_phi"),getattr(ev, "LepW_mass"))
        LepW_Mt=LepW.Mt()


        if (typ=="test"  and i<1000000): continue
        if (typ=="train" and i>=1000000): break
        if (i%10000) == 0: print "%i entries of %i of sample %s loaded"%(i,nEntries,filename)

        #First find if the event
        if ev.nLepSel < 3 or ev.nOSSF_3l > 1: continue #Require at least two OSSF pairs so there can be ambiguity 


        #Convert from the W,Z1,Z2 basis to the correct,lW,lZ basis
        if getattr(ev, "LepW_pdgId")*getattr(ev, "LepZ1_pdgId") < 0:
            lW = "LepW" 
            lZ = "LepZ1"
            lneutral = "LepZ2"
        else:
            lW = "LepW" 
            lZ = "LepZ2"
            lneutral = "LepZ1"

        #Now add variables that are lepton dependant
        tv = []
        for l in [lW,lZ,lneutral]:
            for var in ["pt", "phi", 'eta', "mass", 'conePt', 'dxy', 'dz', 'mva', 'jetDR', 'ptratio', "sip3d", "miniRelIso"]:
               tv.append(getattr(ev, l+"_" + var)) 
        
        #And finally some global event variables
        for gvar in ["mll_3l","mT_3l", 'deltaR_WZ', 'wzBalance_pt', 'wzBalance_conePt', 'm3Lmet']:
            tv.append(getattr(ev, gvar))


        #Add then to the return vector
        v.append(tv)
        
        #And remember to extract the tag of the event
        tags.append(abs(getattr(ev, "LepW_mcMatchId"))==24)

    nmv = np.asarray(v)
    tmv = np.asarray(tags)

    #Creating the weight vector for the loss function
    wv = []
    
    return nmv, tmv


# Training data + tags
x_train, y_train = getNumpyArrayFromFile("./evVarFriend_WZTo3LNu_amcatnlo.root", "train")

# Testing data + tags
x_test, y_test = getNumpyArrayFromFile("./evVarFriend_WZTo3LNu_amcatnlo.root", "test")

#To avoid class imbalance
class_weight = {0: 1./len(np.where(y_train==0)),
                1: 1./len(np.where(y_train==1))}

#Number of non-zero elements in training data
print(np.count_nonzero(y_train))


#NN structure
model = Sequential()
model.add(Dense(42, input_dim=42, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(42, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(42, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(42, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(42, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adamax',
              metrics=['binary_accuracy'],
              weighted_metrics=['accuracy'])

hist = model.fit(x_train, y_train,
                 epochs=75, batch_size=64,
                 validation_data=(x_test,y_test),
                 class_weight=class_weight)


scoreTest   = np.asarray(model.predict(x_test , batch_size=64))
scoreTrain  = np.asarray(model.predict(x_train, batch_size=64))

#Save the weights in a file
model.save("pesosfinal.h5")

#To visualize the shape of your NN
plot_model(model,to_file="model.dot")

#To see the loss function by epoch
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss function (Test)','Loss function (Training)'], loc='upper left')
plt.show()
plt.clf()

#To compare the output distributions

yieldsBad, edges = np.histogram(scoreTest[np.where(y_test==0)[0]], np.linspace(0.7,1,100))
centers = (edges[:-1] + edges[1:])/2.
yieldsBadErr = np.sqrt(yieldsBad)/sum(yieldsBad*1.)
yieldsBad = yieldsBad/sum(yieldsBad*1.)
plt.errorbar(centers, yieldsBad, yerr=yieldsBadErr, label="Bad Match, Test", fmt='o', color="r")

yieldsGood, edges = np.histogram(scoreTest[np.where(y_test==1)[0]], np.linspace(0.7,1,100))
centers = (edges[:-1] + edges[1:])/2.
yieldsGoodErr = np.sqrt(yieldsGood)/sum(yieldsGood*1.)
yieldsGood = yieldsGood/sum(yieldsGood*1.)
plt.errorbar(centers, yieldsGood, yerr=yieldsGoodErr, label="Good Match, Test", fmt='o', color="b")

plt.hist(scoreTrain[np.where(y_train==0)[0]], bins=np.linspace(0.7,1.,100), alpha= 0.3, label="Bad Match, Train", color="r", weights=np.ones_like(scoreTrain[np.where(y_train==0)[0]])/len(scoreTrain[np.where(y_train==0)[0]]) )
plt.hist(scoreTrain[np.where(y_train==1)[0]], bins=np.linspace(0.7,1.,100), alpha= 0.3, label="Good Match, Train", color="b", weights=np.ones_like(scoreTrain[np.where(y_train==1)[0]])/len(scoreTrain[np.where(y_train==1)[0]]))
plt.xlabel("NN Score")
plt.ylabel("Frequency")
plt.legend(loc="best")
plt.show()
plt.clf()

 
nnbins = np.linspace(0.,1,10000)
mmbins = []
for p in nnbins:
    #print len(np.where(scoreTest[np.where(y_test==0)[0]] < p)[0]), len(np.where(scoreTest[np.where(y_test==1)[0]] > p)[0]), (1.*len(y_test))
    mmbins.append(1.-(len(np.where(scoreTest[np.where(y_test==0)[0]] < p)[0]) + len(np.where(scoreTest[np.where(y_test==1)[0]] > p)[0]))/(1.*len(y_test)))

zmin = np.min(mmbins)

for i in range(len(mmbins)):
      if mmbins[i]==zmin: ymin=nnbins[i]

plt.plot(nnbins,mmbins)
plt.xlabel("NN cut")
plt.ylabel("Mistag rate")
plt.show()

print "Original mistag rate: ", len(np.where(y_test==0)[0])/(1.*len(y_test))
print "Actual mistag rate: ", zmin
print "NN cut: ", ymin

