classifiaction_part2:classifiaction
	python Classification_Part2.py

classifiaction:whatIfAnalysis
	python Classification_Part1.py

whatIfAnalysis:prediction
	python Prediction_whatIfAnalysis.py

prediction: dataPreProcessing
	python prediction.py

dataPreProcessing:dataDownloading
	python DataPreProcessing.py
	
dataDownloading:
	python DataDownloading.py
	