
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
parser.add_argument("-n_epochs",type=int,default=-1);
parser.add_argument("-raiseExc",type=int,default=0);
parser.add_argument("-practice",type=int,default=0);
parser.add_argument("-dataLoc",type=str,default="/tmp/f002nb9/oco/mnist.pickle")

opt = parser.parse_args()

data = prepData(opt.dataLoc)

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

def alreadySeen(results, criteria):
	seen = 0
	for r in results:
		match = True
		for crit in criteria:
			val = crit[1]
			if crit[0] == "batch_size" and val == -1: val = 60000
			if r[crit[0]] != val:
				match = False
				break
		if match:
			seen += 1

	return seen

def mainHypSearch(descType, n_epochs=(10000 if opt.n_epochs == -1 else opt.n_epochs)):

	results, succ = get_results()
	while not succ:
		results, succ = get_results()

	for batch_size in [1, -1]:
		for projDim in [1,10,-1,100][::opt.bw]:
			for regLamb in [0,0.5]:

				if alreadySeen(results, (("descType",descType),("batch_size",batch_size),("projDim",projDim),("regLamb",regLamb))):
					continue;

				try:
					gradient_descent(data, opt, lrStrat="epochPro", n_epochs=(n_epochs if batch_size != -1 else int(n_epochs/10)), batch_size=batch_size, regLamb=regLamb, fake=False,
								 easyBin=False, projDim=projDim, quickie=opt.quickie, descType=descType)
				except Exception as e:
					print("oh bother",opt.batch_size, projDim, regLamb)
					if opt.raiseExc:
						raise e

def newtonHypSearch(descType, n_epochs=(10000 if opt.n_epochs == -1 else opt.n_epochs)):

	results, succ = get_results()
	for projDim in [1,10,100,-1][::opt.bw]:
		for regLamb in [0,0.2,1]:

			if alreadySeen(results, (("descType", descType),("projDim", projDim), ("regLamb", regLamb))):
				continue;

			try:
				gradient_descent(data, opt, lrStrat="epochPro", n_epochs=n_epochs, batch_size=opt.batch_size, regLamb=regLamb, fake=False,
							 easyBin=False, projDim=projDim, quickie=opt.quickie, descType=descType,gamma=opt.gamma)
			except:
				pass


def expHypSearch(descType, n_epochs=(100000 if opt.n_epochs == -1 else opt.n_epochs)):

	results, succ = get_results()
	for projDim in [1,10,100,500]:
		if alreadySeen(results, (("descType", descType), ("projDim", projDim))):
			continue;
		try:
			gradient_descent(data, opt, lrStrat="epochPro", n_epochs=n_epochs, batch_size=opt.batch_size, regLamb=0, fake=False,
						 easyBin=False, projDim=projDim, quickie=opt.quickie, descType=descType)
		except:
			print("oh bother",opt.batch_size, projDim)

main()
