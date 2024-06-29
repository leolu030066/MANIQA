

class GFIQA(Dataset):
    def __init__(self, dis_path, csv_file_name, transform=None):
        super(GFIQA, self).__init__()
        self.dis_path = dis_path
        self.csv_file_name = csv_file_name
        self.transform = transform
        
        data = pd.read_csv(self.csv_file_name)
        all_img_names = data['img_name'].tolist()
        all_scores = data['mos'].astype(float).tolist()
        
        filtered_img_names = []
        filtered_scores = []
        for img_name, score in zip(all_img_names, all_scores):
            img_path = os.path.join(self.dis_path, img_name)
            if os.path.exists(img_path):
                filtered_img_names.append(img_name)
                filtered_scores.append(score)
        
        self.score_data = np.array(filtered_scores)
        self.score_data = self.normalize(self.score_data).reshape(-1, 1)
        
        self.data_dict = {'d_img_list': filtered_img_names, 'score_list': self.score_data}


    def normalize(self, data):
        data_range = np.max(data) - np.min(data)
        return (data - np.min(data)) / data_range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_path = os.path.join(self.dis_path, d_img_name) 
        d_img = cv2.imread(d_img_path, cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))  

        score = self.data_dict['score_list'][idx]
        sample = {'d_img_org': d_img, 'score': score}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
