class EntData():
    def __init__(self, train_file, dev_file, test_file):
        """ Constructor method. Loads and prepares data.
        Inputs
        -----------
        train_file  - str. Path to the training file.
        dev_file    - str. Path to the dev file.
        test_file   - str. Patht to the test file.
        """
        self.train_data, self.labels = self.data_prep(train_file)
        self.dev_data, _ = self.data_prep(dev_file, label_set = self.labels)
        self.test_data, _ = self.data_prep(test_file, label_set = self.labels)



    def data_prep(self, file_name, label_set = None):
        """ Read data from the file_name. Files are supposed to be 
        constructed in the .conll format, i.e, a line is '<token>\t<label>'
        Inputs
        -----------
        file_name   - str. Path to the file from which the data
                        needs to be extracted.
        label_set   - list[str]. List of all distinct labels in the data.
                        Default: None

        Outputs
        ----------
        processed_data - dict{List}. Processed data. The words are combined into
                            a list for a particular sentence. This list is stored
                            in the 'inp' key of the dictionary. The labels are 
                            stored in a parallel list with the labels converted to 
                            suitable integers.
        """
        with open(file_name) as f:
            data = f.readlines()

        # Gather the label set if not computed already
        if label_set is None:
            label_set = []
            for line in data:
                if line in ["\n","\t\n"]:    #Skip line breaks
                    continue 
                lab = line.strip("\n").split("\t")[1]   #Extracting the label
                if lab =='':
                    print(list(line))
                    
                # Collecting distinct labels 
                if lab not in label_set:
                    label_set.append(lab)
        
        processed_data = {'inp' : [] , 'label' : [] }   # Dictionary to store the processed data
        sent = []   # A list to maintain the sentence being processed
        label_sent = [] # A list to maintain the label for the sentence being processed
        for line in data:
            if line in ["\n","\t\n"]:   # Skip line breaks
                processed_data['inp'].append(sent)
                processed_data['label'].append(label_sent)
                sent = []
                label_sent = []
            else:
                data = line.strip("\n").split("\t")
                sent.append(data[0])
                label_sent.append(label_set.index(data[1]))
                


        return processed_data, label_set

