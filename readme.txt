(A)
   run pearson.py to see the correlation among X-Y and Y-Y base which we model our condition random field

(B)
   run main.py  to 
	(1) compute log loss using five-fold cross validation on quiztrain.csv
	(2) compute log loss on test dataset (e.g. quiztest.csv)
   
   feel free to change the model structures that are defined in XY and YY in main.py
   The following is our setting
        isolated y's
        12
	13
	20

	chain without x's

	11----17----18-----19


	chain with x's

	5    9    2    3    2    4
	 \  /      \  /      \  /
	  16  ----- 15 -------14
	
	corresponding code in python 
	     XY = [
		 {12:[x for x in range(1, 11)]},
		 {13:[x for x in range(1, 11)]},
		 {20:[x for x in range(1, 11)]},
		 {16:[5,9], 15:[2,3], 14:[2,4]},
		 {},
	     ]
	     YY = [
		[12],
		[13],
		[20],
		[16,15,14],
		[11,17,18,19]
	     ]   

	our codes were developped based on the crf.py by Graham Neubig