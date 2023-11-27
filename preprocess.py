import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

def load_piu_data():
    # load inertet addiction surveys data from a file
    file_name = "./datasets/piu_addiction.csv"
    data = pd.read_csv(file_name, na_values = ' ')

    # remove irrelevant columns
    # columns ID, TEMPS1 to TEMPS36, NKP, PI, SPO, CutOff4950, EnergetskoP2 i Temper_bin are helpers and not part of of the analysis
    temp_cols=[]
    for col in data.columns:
        if col.startswith('TEMPS'):
            temp_cols.append(col)
    out_cols = ['ID','NKP','PI','SPO', 'CutOff4950', 'EnergetskoP2', 'Temper_bin','SkolaPoRegionu','SkolaPoTipu','Komunikacija1','Komunikacija2','Komunikacija3','Komunikacija4','ImaKomp','ZadovoljanPristupom','BrojaMailovaPrim','BrojaMailovaPosl','ZaStaMail','ImaNemaFB','PusacKolikoGodina','PusacKolikoCigareta','Godine','DaLiSeDrogira']
    for col in out_cols:
        temp_cols.append(col)
    data = data.drop(temp_cols, axis=1)
    
    # rename column
    features = {
        'CutOff3940':'Cutoff_Class',
        'PUI':'PIU',
        'PUIcutoff':'PIUcutoff'
    }
    data.rename(columns = features, inplace=True)

    # columns Internet1 to Internet18 are helpers and not part of the analysis
    internet_cols=[]
    for col in data.columns:
        if col.startswith('Internet'):
            internet_cols.append(col)
    data = data.drop(internet_cols, axis=1)

    # rename columns from Serbian to English
    features = {
        'Pol':'Gender',
        'Uspeh':'Achievement',
        'EkonomskiPolozaj':'Economic status',
        'KolikoDugo':'Internet Use (in years)',
        'KolikoNedeljno':'Internet Use (hours per week)',
        'KolikNajduze':'Internet Use (hours per day)',
        'Predhodnih6meseci':'Internet Use (in holiday)',
        'DaMozeDaLiBi':'Attitude about time on the Internet',
        'Sadrzaj1':'Politics',
        'Sadrzaj2':'Business',
        'Sadrzaj3':'Sports',
        'Sadrzaj4':'Computers and technology',
        'Sadrzaj5':'Arts and culture',
        'Sadrzaj6':'Education',
        'Sadrzaj7':'Pop culture',
        'Sadrzaj8':'Pornography',
        'Sadrzaj9':'Music',
        'Sadrzaj10':'Travel/tourism',
        'Sadrzaj11':'Health and medicine',
        'Sadrzaj12':'Science',
        'Sadrzaj13':'Religion',
        'Aktivnost1': 'Communication by e-mail',
        'Aktivnost2':'Social networks',
        'Aktivnost3':'Communication on the forum',
        'Aktivnost4':'Communication on the blog',
        'Aktivnost5':'Targeted Internet search',
        'Aktivnost6':'Surfing',
        'Aktivnost7':'Expert Advice',
        'Aktivnost8':'Search for favorite websites',
        'Aktivnost9':'Reading the news',
        'Aktivnost10':'Online games',
        'Aktivnost11':'Reading and downloading books and texts',
        'Aktivnost12':'Downloading music and movies',
        'Aktivnost13': 'Internet for school',
        'Aktivnost14':'Online courses',
        'DaLiSvakodnevnoFb':'Everyday FB use',
        'BrojaSatiFB':'Average time spent on FB',
        'FBigraIgrice':'FB use –gaming',
        'FBcetuje':'FB use – chatting',
        'FBgrupe':'FB use – visiting groups',
        'FBcitaPostove':'FB use - reading posts',
        'FBpiseStatuse':'FB use - publishing statuses',
        'FBdeliMuzikuFotografijeIsl':'FB use - sharing music, photos etc.',
        'FizAkt1':'Sports – days in a  week',
        'FizAkt2':'Sports – intensity',
        'FizAkt3':'Sports – in minutes',
        'EnergetskoP1':'Energy drinks',
        'EnergetskoP2': 'Energy drinks (ml)',
        'Grickalice':'Fast Food',
        'Pusac':'Smoker',
        'Kafa_bin':'Coffee',
        'Alkohol_bin':'Alcohol',
        'Depresivan':'Drepressive temperament',
        'Ciklotimicni':'Cyclothymic temperament',
        'Hipertimicni':'Hyperthymic temperament',
        'Iritabilni':'Irritable temperament',
        'Anksiozni':'Anxiety temperament'
    }
    data = data.rename(columns=features)

    return data

def determine_bool_columns(data):
    # determine columns taht have only binary values [0, 1]
    bool_columns = [col for col in data if data[col].dropna().value_counts().index.isin([0,1]).all()]
    return bool_columns

def process_standardization(data):
    # columns that should not be scaled (columns with binary values and output columns)
    no_process = determine_bool_columns(data)
    no_process.append('PIU')
                                                             
    to_process = data.columns.difference(no_process)
    
    # robust scaling
    standard_scaler = StandardScaler()
    new_data = data.copy()
    new_data[to_process] = standard_scaler.fit_transform(new_data[to_process])
                                                             
    return new_data

def process_columns_with_nan_values(data):

    # Cutoff_Class column has 104 rows with NaN value, aproximately 5% of the data set
    # delete rows where Cutoff_Class column has NaN value
    data = data.dropna(subset=['Cutoff_Class', 'PIU'])

    # for rest of the data use knn imputer to replace missing values
    imputer = KNNImputer(n_neighbors=10)
    new_data = imputer.fit_transform(data)

    return pd.DataFrame(new_data, columns=data.columns)

def process_outliers(data):
    print("Original number of rows: {}".format(len(data)))
    isolation_forest = IsolationForest(n_estimators = 100, contamination = 0.05, max_samples = 'auto')
    outlier_prediction = isolation_forest.fit_predict(data.values)
    print("Number of normal rows detected: {}".format(outlier_prediction[outlier_prediction  == 1].sum()))
    print("Number of outliers detected: {}".format(abs(outlier_prediction[outlier_prediction == -1].sum())))
    data_no_outliers = data.copy()
    data_no_outliers['Is_Outlier'] = outlier_prediction
    data_no_outliers.drop(data_no_outliers[(data_no_outliers['Is_Outlier'] == -1)].index, axis = 0, inplace=True)
    data_no_outliers = data_no_outliers[data_no_outliers.columns.difference(['Is_Outlier'])]
    print('Number of rows after eliminating outliers: ' + str(len(data_no_outliers)))
    return data_no_outliers




    

    

    