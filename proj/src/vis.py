

imgOutPath = "./img"
import numpy as np
import matplotlib.pyplot as plt
def plot_results(d, title):
	allLosses, allAccsTrain, allAccsTrainSimple, \
	allAccsTest, allAccsTestSimple = d["wholeDSLosses"], d["accTrain"], d["accTrainSimple"], d["accTest"], d["accTestSimple"]
	plt.figure(dpi=80)
	ax = plt.gca()
	ls = ax.plot(allLosses ,label="loss")
	twinx = ax.twinx()
	tr = twinx.plot(allAccsTrainSimple ,color="green" ,label="weighted train accuracy" ,linestyle='dashed')
	trh = twinx.plot(allAccsTrain ,label="harmonic train accuracy" ,color="green")
	ts = twinx.plot(allAccsTestSimple ,color="red" ,label="weighted test accuracy" ,linestyle='dashed')
	tsh = twinx.plot(allAccsTest ,label="harmonic test accuracy" ,color="red" ,)
	lns = ls + trh + tr + tsh + ts
	labs = [l.get_label() for l in lns]
	ax.legend(lns, labs ,loc="lower right")
	ax.set_xlabel("Epoch");
	ax.set_ylabel("Loss")
	twinx.set_ylabel("Accuracy")
	plt.title(title)
	plt.savefig(imgOutPath + "/" +d["outPath"]);


def stoch_vs_not(df):
	# [x[0]-x[1] for x in [["accTest_final",

	for descType in ["gradDesc"]:
		algoDf = df[df.descType == descType].groupby(["regLamb","projDim","n_epochs"]).mean().reset_index()
		dfComp = algoDf.groupby(["regLamb", "projDim"]).agg(list).reset_index()
		dfComp["timePerEpoch"] = dfComp.apply(lambda row: [row.runtime[i]/row.n_epochs[i] for i in range(len(row.n_epochs))],axis=1)

		plt.figure()
		ax = plt.gca()
		times = ax.bar(np.arange(len(dfComp)),dfComp.timePerEpoch.apply(lambda x: x[0]/x[1]),width=1/3,color="green",label= "Time per Epoch")
		ax.set_ylabel("Ratio of Time per Epoch")

		twinx = ax.twinx()
		accs = twinx.bar(np.arange(len(dfComp))+1/3,dfComp.accTest_final.apply(lambda x: x[0]/x[1]),width=1/3,color="red",label="Test Accuracy")
		twinx.set_ylabel("Ratio of Test Accuracy")

		twinx.plot([0,len(dfComp)], [1,1], linestyle='dashed',color="red",alpha=0.5)
		ax.plot([0,len(dfComp)], [1,1], linestyle='dashed',color="green",alpha=0.5)

		lns = [times, accs]
		labs = [l.get_label() for l in lns]
		ax.legend(lns, labs)
		plt.title("Comparison of Time and Accuracy in Stochastic vs Non-Stochastic GD")
		plt.show()

		plt.savefig(imgOutPath + "/ " +"stochGradComp.png");



def other_vs_sgd(df):
	# [x[0]-x[1] for x in [["accTest_final",

	for descType in ["newtonONS"]:
		algoDf = df[df.descType.isin([descType,"gradDesc"])].groupby(["descType","regLamb","projDim","n_epochs"]).mean().reset_index()
		algoDf = algoDf[algoDf.accTestSimple_final > 0.5]
		dfComp = algoDf.groupby(["regLamb","projDim","n_epochs"]).agg(list).reset_index()
		dfComp = dfComp[dfComp.descType.apply(lambda x: len(x)) ==2]
		dfComp["timePerEpoch"] = dfComp.apply(lambda row: [row.runtime[i]/row.n_epochs for i in range(2)],axis=1)

		plt.figure()
		ax = plt.gca()
		times = ax.bar(np.arange(len(dfComp)),dfComp.timePerEpoch.apply(lambda x: x[0]/x[1]),width=1/3,color="green",label= "Time per Epoch")
		ax.set_ylabel("Ratio of Time per Epoch")
		ax.plot([0,len(dfComp)], [1,1], linestyle='dashed',color="green",alpha=0.5)

		twinx = ax.twinx()
		accs = twinx.bar(np.arange(len(dfComp))+1/3,dfComp.accTest_final.apply(lambda x: x[0]/x[1]),width=1/3,color="red",label="Test Accuracy")
		twinx.set_ylabel("Ratio of Test Accuracy")

		twinx.plot([0,len(dfComp)], [1,1], linestyle='dashed',color="red",alpha=0.5)
		ax.plot([0,len(dfComp)], [1,1], linestyle='dashed',color="green",alpha=0.5)

		lns = [times, accs]
		labs = [l.get_label() for l in lns]
		ax.legend(lns, labs,loc="upper right")
		ax.set_zorder(-1)
		twinx.set_zorder(-1)

		plt.title("Comparison of Time and Accuracy in " + descType + " vs Projected Stochastic GD")
		plt.show()

		plt.savefig(imgOutPath + "/ " +"stochGradComp.png");




def all_hist(df):


	for descType in ["newtonONS"]:
		algoDf = df[df.descType.isin([descType,"gradDesc"])].groupby(["descType","regLamb","projDim","n_epochs"]).mean().reset_index()
		algoDf = algoDf[algoDf.accTestSimple_final > 0.5]
		dfComp = algoDf.groupby(["regLamb","projDim","n_epochs"]).agg(list).reset_index()
		dfComp = dfComp[dfComp.descType.apply(lambda x: len(x)) ==2]
		dfComp["timePerEpoch"] = dfComp.apply(lambda row: [row.runtime[i]/row.n_epochs for i in range(2)],axis=1)

		plt.figure()
		ax = plt.gca()
		times = ax.bar(np.arange(len(dfComp)),dfComp.timePerEpoch.apply(lambda x: x[0]/x[1]),width=1/3,color="green",label= "Time per Epoch")
		ax.set_ylabel("Ratio of Time per Epoch")
		ax.plot([0,len(dfComp)], [1,1], linestyle='dashed',color="green",alpha=0.5)

		twinx = ax.twinx()
		accs = twinx.bar(np.arange(len(dfComp))+1/3,dfComp.accTest_final.apply(lambda x: x[0]/x[1]),width=1/3,color="red",label="Test Accuracy")
		twinx.set_ylabel("Ratio of Test Accuracy")

		twinx.plot([0,len(dfComp)], [1,1], linestyle='dashed',color="red",alpha=0.5)
		ax.plot([0,len(dfComp)], [1,1], linestyle='dashed',color="green",alpha=0.5)

		lns = [times, accs]
		labs = [l.get_label() for l in lns]
		ax.legend(lns, labs,loc="upper right")
		ax.set_zorder(-1)
		twinx.set_zorder(-1)

		plt.title("Comparison of Time and Accuracy in " + descType + " vs Projected Stochastic GD")
		plt.show()

		plt.savefig(imgOutPath + "/ " +"stochGradComp.png");




def all_best_runs(df):
	accTests = df.iloc[df.groupby("descType").idxmax().accTest_final][["accTest", "descType"]]
	accTests.apply(lambda row: plt.plot(np.arange(len(row.accTest)) / len(row.accTest), row.accTest,label=row.descType), axis=1)
	plt.legend()
	plt.title("Best performances for each algorithm")
	plt.ylabel("Weighted Accuracy")
	plt.xlabel("Epoch (relative)")
	plt.xticks([])
	plt.show()


