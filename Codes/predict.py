import numpy as np
import pandas as pd 
from autogluon.text import TextPredictor
from Bio import SeqIO
import argparse
import json

def Csv_to_json(input_path,result_path):
    '''
    This function takes two arguments input_path and result_path. 
    It loads the csv file from input_path, 
    it converts the data from csv to json format and 
    writes it to a file named results.json in the result_path.
    '''
    csvData = pd.read_csv(input_path, header = 0)  
    columns = csvData.columns.tolist()
    dimensionoutPut = {}
    for index in range(len(csvData)):
        outPut = {}
        for col in columns:
            outPut[col] = str(csvData.loc[index, col])
        dimensionoutPut[str(index+1)] = outPut
    jsonData = json.dumps(dimensionoutPut,indent=4) 
    with open(result_path + "/results.json", 'w') as jsonFile:   
        jsonFile.write(jsonData)


def main():
    seq_path = args.input_path

    # loads a pre-trained model called "model_9" using the TextPredictor class from autogluon library.
    predictor = TextPredictor.load("./model_9")

    # reads fasta sequences from the input_path and filters the sequences which are 51 in length and have K at position 25.
    detected_seq = {"name": [], "seq": []}
    for fa in SeqIO.parse(seq_path, "fasta"):
        seq = fa.seq.upper()
        if len(seq) == 51 and seq[25] == "K":
            detected_seq["name"].append(fa.name)
            detected_seq["seq"].append([" ".join(seq)])

    if len(detected_seq["seq"]) > 0:

        # ses the loaded model to predict the lactylation of the filtered sequences.
        input_seqs = detected_seq["seq"]
        predict = np.array(predictor.predict_proba(pd.DataFrame(input_seqs).rename(
        columns={0: "sequence"})))

        lactylation = np.argmax(predict, axis=1).flatten()
        lactylation = ["Yes" if x == 1 else "No" for x in lactylation]
        results = dict()
        results["Name"] = np.array(detected_seq["name"])
        results["Position"] = [26] * len(results["Name"])
        results["Residue"] = ['K'] * len(results["Name"])
        results["Kla Probability"] = np.array([format(x,".2%") for x in predict[:, 1]])
        results["Lactylation"] = lactylation

        # creates a dataframe from the results and writes it to a csv file named results.csv in the result_path directory.
        df = pd.DataFrame(results)
        df.to_csv(args.result_path + "/results.csv",index=False,encoding="utf_8_sig")

        # calls the Csv_to_json function to convert the csv results to json format and write it to the result_path directory.
        Csv_to_json(args.result_path + "/results.csv",args.result_path)

    # If no sequences are filtered, it writes an empty dataframe to the csv and json files.    
    else:
        results = dict()
        results["Name"] =  []
        results["Position"] =  []
        results["Residue"] =  []
        results["Kla Probability"] =  []
        results["Lactylation"] =  []
        df = pd.DataFrame(results)
        df.to_csv(args.result_path + "/results.csv",index=False,encoding="utf_8_sig")
        Csv_to_json(args.result_path + "/results.csv",args.result_path)

if __name__ == "__main__":

    # sets up the command line argument parser and assigns default values to the input_path and result_path.
    parser = argparse.ArgumentParser(description="Kla")
    parser.add_argument('-input_path',default='./example.fasta')
    parser.add_argument('-result_path',default='./') 
    args = parser.parse_args()    

    # starts the main function when the script is run.
    main()