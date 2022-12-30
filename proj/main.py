
try:
	from .src import *
except:
	from src import *;

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-reset', type=int, default=0)
parser.add_argument("-quickie",type=int,default = 0)
parser.add_argument("-descType",type=str);
parser.add_argument("-gamma",type=float,default=1/8);
parser.add_argument("-projDim",type=float,default=1);
parser.add_argument("-purge",type=float,default=0);
parser.add_argument("-bw",type=int,default=1);
parser.add_argument("-batch_size",type=int,default=1);
# parser.add_argument("-n_epochs",type=int,default=100000);


opt = parser.parse_args()


def main():

	setupFolders(opt.purge)


	if "gradDesc" in opt.descType:
		mainHypSearch("gradDesc")
	if "mirrDesc" in opt.descType:
		mainHypSearch("mirrDesc")
	if "expGrad" in opt.descType:
		mainHypSearch("expGrad")
	if "adaGrad" in opt.descType:
		mainHypSearch("adaGrad")
	if "newtonONS" in opt.descType:
		newtonHypSearch("newtonONS")
	if "randExp" in opt.descType:
		expHypSearch("randExp")
	if "bandExp" in opt.descType:
		expHypSearch("bandExp")

def mainHypSearch(descType, n_epochs=500):

	for batch_size in [4,16,64,-1]:
		for projDim in [1,10,100,-1]:
			for regLamb in [0,0.2,1]:
				try:
					gradient_descent(data, lrStrat="epochPro", n_epochs=n_epochs, batch_size=batch_size, regLamb=regLamb, fake=False,
								 easyBin=False, projDim=projDim, quickie=opt.quickie, descType=descType)
				except:
					print("oh bother",batch_size, projDim, regLamb)

def newtonHypSearch(descType, n_epochs=5000):

	for batch_size in [64,32,-1][::opt.bw]:
		for projDim in [1,10,100,-1]:
			for regLamb in [0,0.2,1]:
				try:
					gradient_descent(data, lrStrat="epochPro", n_epochs=n_epochs, batch_size=batch_size, regLamb=regLamb, fake=False,
								 easyBin=False, projDim=projDim, quickie=opt.quickie, descType=descType,gamma=opt.gamma)
				except:
					pass



def expHypSearch(descType, n_epochs=100000):

	for batch_size in [64]:
		try:
			gradient_descent(data, lrStrat="epochPro", n_epochs=n_epochs, batch_size=batch_size, regLamb=0, fake=False,
						 easyBin=False, projDim=opt.projDim, quickie=opt.quickie, descType=descType)
		except:
			print("oh bother",batch_size, opt.projDim)

main()
