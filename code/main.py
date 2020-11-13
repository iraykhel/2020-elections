import requests
import pprint
from sqlite import *
import csv
import traceback
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics

class Elections:
    def __init__(self):
        self.db = SQLite()
        self.db.connect('elections')
        self.state_codes = self.read_in_states()

    def read_in_states(self):
        state_codes = {}
        f = open('data/states.txt', 'r')
        lines = f.readlines()
        for line in lines:
            state, code = line.split('\t')
            state_codes[code.strip()] = state.strip()
        return state_codes

    def create_db(self):
        self.db.create_table('results_2020',"id integer primary key autoincrement, county text, state text, biden, trump, total",drop=False)
        self.db.create_table('results_2016',
                             "id integer primary key autoincrement, county text, state text, clinton, trump, total",
                             drop=True)
        dct = elections.get_census_quickfacts('iowa','obrien')
        fields = ['id integer primary key autoincrement','state text','county text']
        for field in dct:
            fields.append("`"+field+"` NUMERIC")
        fields = ','.join(fields)
        print(fields)
        self.db.create_table('census_quickfacts', fields, drop=False)


    def dd_2020_download_data(self,state_code):
        cnt = 0
        url = 'https://embeds.ddhq.io/api/v2/2020general_results/2020general_'+state_code.lower()
        res = requests.get(url).json()

        # pprint.pprint(res['data'][0])
        for race in res['data']:
            if race['office'] == 'President':
                break

        candidates = race['candidates']
        for entry in candidates:

            if entry['last_name'] == 'Trump':
                trump_id = str(entry['cand_id'])
            if entry['last_name'] == 'Biden':
                biden_id = str(entry['cand_id'])
        # print(trump_id,biden_id)

        if 'countyResults' in race:
            counties = race['countyResults']['counties']
        else:
            counties = race['vcuResults']['counties']
        for entry in counties:


            name = entry['county']
            if name is None:
                print(state_code+" has a None county", entry)
                continue
            votes = entry['votes']
            total = 0

            for cand_id, vote_count in votes.items():
                if cand_id == trump_id:
                    trump_cnt = vote_count
                if cand_id == biden_id:
                    biden_cnt = vote_count
                total += vote_count
            # print(name, trump_cnt, biden_cnt, total)
            precincts = entry['precincts']
            if precincts['reporting'] != precincts['total']:
                print("Incomplete result for " + state_code + " county " + name, precincts)
            if biden_cnt == 0 or trump_cnt == 0 or total == 0:
                print("Bad result for "+state_code+" county "+name)
            else:
                db_entry = {'county':name.replace('\'',''),'state':state_code.upper(),'biden':biden_cnt,'trump':trump_cnt,'total':total}
                cnt += 1
                self.db.insert_kw('results_2020',**db_entry)
        self.db.commit()
        return cnt

    def download_2020_results(self):
        for state_code in self.state_codes:
            self.dd_2020_download_data(state_code)



    def read_in_2016_results(self):
        reader = csv.reader(open('data/2016_US_County_Level_Presidential_Results.csv'))
        next(reader)
        for row in reader:
            county = row[-2]
            trump = row[2]
            clinton = row[1]
            total = row[3]
            state_code = row[-3]
            county_words = county.split(" ")
            if 'County' in county_words:
                county_words.remove('County')
                print(county_words)
            if 'Parish' in county_words:
                county_words.remove('Parish')
            name = ' '.join(county_words)
            db_entry = {'county': name.replace('\'', ''), 'state': state_code.upper(), 'clinton': int(float(clinton)), 'trump': int(float(trump)), 'total': int(float(total))}
            self.db.insert_kw('results_2016', **db_entry)
            # print(state_code, county,trump,clinton,total)
        self.db.commit()


    def get_census_quickfacts(self,state_code,county):
        state = self.state_codes[state_code]
        county = county.lower().replace(' ','').replace('.','').replace('-','')
        state = state.lower().replace(' ','')
        if state == 'louisiana':
            county_designation = 'parish'
        else:
            county_designation = 'county'

        found = False
        if county == 'districtofcolumbia':
            location = 'DC'
            found = True
        elif 'city' in county and county not in ['jamescity','charlescity']:
            location = county+state+county_designation
            found = True
        elif state == 'newyork':
            borough_map = {
                'bronx':['bronx'],
                'brooklyn':['kings'],
                'manhattan':['newyork'],
                'queens':['queens'],
                'statenisland':['richmond']
            }
            for borough,counties in borough_map.items():
                if county in counties:
                    location = county + county_designation + borough + 'borough' + state
                    found = True
                    break

        if not found:
            location = county + county_designation+state

        try:
            lines = open('data/'+county+"_"+state_code.lower()+".csv",'r')
            print("loaded from file")
        except:
            url = 'https://www.census.gov/quickfacts/fact/csv/'+location
            print(url)
            print(location)
            res = requests.get(url)
            lines = res.content.decode('utf-8').split("\n")

        try:
            reader = csv.reader(lines)
            next(reader)
            rv = {}
            for row in reader:
                # print('r',row)
                header = row[0].strip()
                if header == 'FIPS Code':
                    break
                value = row[-2].strip()
                if value[-1] == '%':
                    value = value[:-1]
                if value[0] == '$':
                    value = value[1:]

                if value in ['Z','F']:
                    value = 0
                elif value in ['-','FN','N','NA','S','X','D']:
                    value = ''
                else:
                    value = value.replace(',','')
                    value = float(value)


                # print(header, "|", value)
                rv[header] = value
            return rv
        except:
            print("Couldn't get census for "+state+" "+county, traceback.format_exc())
            return None

    def get_census_all(self):
        counties = self.db.select("SELECT county, state FROM results_2020 ORDER BY state ASC")
        for idx,row in enumerate(counties):
            county = row[0].lower()
            state_code = row[1]

            check = self.db.select("SELECT * FROM census_quickfacts WHERE county='"+county+"' AND state='"+state_code+"'")

            if len(check) == 0:
                print(state_code, county)
                cens = self.get_census_quickfacts(state_code,county)
                if cens is not None:
                    cens['state'] = state_code
                    cens['county'] = county
                    self.db.insert_kw('census_quickfacts',**cens)
                    self.db.commit()
                    time.sleep(0.3)
            # else:
            #     print("Already done")




    def load_data_from_db(self):
        query = "select c.*, r2020.biden as biden20, r2020.trump as trump20, r2020.total as total20, r2016.clinton as clinton16,r2016.trump as trump16,r2016.total as total16 " \
                "from census_quickfacts as c, results_2020 as r2020, results_2016 as r2016 " \
                "where c.state = r2020.state and c.state = r2016.state and c.county = lower(r2020.county) and c.county = r2016.county"
        self.data = self.db.select(query,dict=True)
        print("Counties",len(self.data))


    def separate_sets(self):
        # control_set = ['vt','nd']
        control_set = []

        test_set = [
            ('az','maricopa'),
            ('az','pima'),
            ('nv','clark'),
            ('pa','allegheny'),
            ('pa', 'montgomery'),
            ('pa', 'philadelphia'),
            ('pa', 'delaware'),
            ('pa', 'bucks'),
            ('ga', 'cobb'),
            ('ga', 'gwinnett'),
            ('ga', 'dekalb'),
            ('ga', 'fulton'),
            ('wi', 'dane'),
            ('wi', 'milwaukee'),
            ('mi', 'wayne'),
            ('mi', 'oakland')
        ]

        control_set_features = []
        control_set_outcome_16 = []
        control_set_outcome_20 = []
        control_set_weights = []

        test_set_features = []
        test_set_outcome_16 = []
        test_set_outcome_20 = []
        test_set_weights = []

        training_set_features = []
        training_set_outcome_16 = []
        training_set_outcome_20 = []
        training_set_weights = []

        self.features = feature_columns = ['Population_change_2010_2019','Under_5','Under_18','Over_65','Women','White','Black','Indian','Asian','Pacific','Hispanic','Veterans','Foreign_born','Houses','Owner_occupied','Rent','Building_permits','Households',
                          'People_per_household','Living_in_same_house_as_1_year_ago','Languages_spoken','Computer','Internet','High_school','Bachelors','Disabled_under_65','No_health_insurance_under_65','In_labor_force','In_labor_force_women',
                          'Sales_retail_per_capita','Commute','Median_household_income','Income','Poverty','Employment','Payroll','Employment_change_2017_2018','Population_density','Land_area']

        non_per_capita_columns = ['Veterans','Houses','Building_permits','Households','Employment','Payroll']
        potential_missing = ['Building_permits','Sales_retail_per_capita']

        skip_cnt = 0
        self.test_set = []
        for entry in self.data:
            skip = False
            state = entry['state'].lower()
            county = entry['county'].lower()


            if state in control_set:
                features_set = control_set_features
                outcome_set_16 = control_set_outcome_16
                outcome_set_20 = control_set_outcome_20
                weight_set = control_set_weights
            elif (state,county) in test_set:
                features_set = test_set_features
                outcome_set_16 = test_set_outcome_16
                outcome_set_20 = test_set_outcome_20
                weight_set = test_set_weights
                self.test_set.append((state,county))
            else:
                features_set = training_set_features
                outcome_set_16 = training_set_outcome_16
                outcome_set_20 = training_set_outcome_20
                weight_set = training_set_weights

            feature_entry = []
            population = float(entry['Population_2019'])
            for column in feature_columns:
                if column in potential_missing:
                    if entry[column] == '':
                        feature_entry.append(0)
                        val = 0
                    else:
                        feature_entry.append(1)
                        val = float(entry[column])
                else:
                    if entry[column] == '':
                        print('skipping country for missing data',state, county, column, population)
                        skip = True
                        break
                    else:
                        val = float(entry[column])

                if column in non_per_capita_columns:
                    val = val / float(population)
                feature_entry.append(val)

            if not skip:
                outcome_2016 = float(entry['clinton16'])/float(entry['total16'])
                outcome_2020 = float(entry['biden20']) / float(entry['total20'])

                features_set.append(feature_entry)
                outcome_set_16.append(outcome_2016)
                outcome_set_20.append(outcome_2020)
                weight_set.append(population)
            else:
                skip_cnt += 1



        # print(len(training_set_features),len(test_set_features),len(control_set_features), skip_cnt)
        self.X_training = np.array(training_set_features)
        self.Y_training_16 = np.array(training_set_outcome_16)
        self.Y_training_20 = np.array(training_set_outcome_20)
        self.weights_training = np.array(training_set_weights)

        self.X_control = np.array(control_set_features)
        self.Y_control_16 = np.array(control_set_outcome_16)
        self.Y_control_20 = np.array(control_set_outcome_20)
        self.weights_control = np.array(control_set_weights)

        self.X_test = np.array(test_set_features)
        self.Y_test_16 = np.array(test_set_outcome_16)
        self.Y_test_20 = np.array(test_set_outcome_20)
        self.weights_test = np.array(test_set_weights)

        # pprint.pprint(self.test_set)


    def set_trainer(self):
        self.trainer = xgb.XGBRegressor(objective='reg:logistic', n_estimators=300, learning_rate=0.1, max_depth=6)

    def cv(self,folds=5):
        kf = KFold(folds, shuffle=True)
        Y = self.Y_training_20
        # X = self.X_training
        X = np.hstack((self.X_training,self.Y_training_16.reshape((len(self.Y_training_16),1))))
        # X = self.Y_training_16.reshape((len(self.Y_training_16),1))
        # s = QuantileTransformer(output_distribution='normal')



        Y_pred_all = np.zeros(len(Y))
        for fold_idx, (train_index,test_index) in enumerate(kf.split(self.X_training,Y)):
            print("Fold",fold_idx)
            X_trainval, X_test = X[train_index], X[test_index]
            Y_trainval, Y_test = Y[train_index], Y[test_index]
            weights_trainval = self.weights_training[train_index]

            X_train, X_val, Y_train, Y_val, weights_train, weights_val = train_test_split(X_trainval, Y_trainval, weights_trainval, test_size=0.33, shuffle=True)
            # X_train = s.fit_transform(X_train)
            # X_val = s.transform(X_val)
            # X_test = s.transform(X_test)

            # trainer.fit(X_train,Y_train,sample_weight=weights_train,eval_set=[(X_val,Y_val)], sample_weight_eval_set=[weights_val], early_stopping_rounds=10, verbose=1, eval_metric='mae')

            X_train = np.vstack((X_train,X_val))
            Y_train = np.hstack((Y_train, Y_val))
            weights_train = np.hstack((weights_train, weights_val))
            self.trainer.fit(X_train,Y_train,sample_weight=weights_train,verbose=1)

            Y_pred = self.trainer.predict(X_test)
            Y_pred_all[test_index] = Y_pred


        mae = metrics.mean_absolute_error(Y,Y_pred_all)
        baseline_pred = np.full(Y.shape, np.average(Y))
        baseline_mae = metrics.mean_absolute_error(Y,baseline_pred)
        print(mae, baseline_mae)

    def train(self):
        Y = self.Y_training_20
        X = np.hstack((self.X_training, self.Y_training_16.reshape((len(self.Y_training_16), 1))))
        # X = self.X_training
        # X = self.Y_training_16.reshape((len(self.Y_training_16), 1))
        W = self.weights_training
        print(X.shape)
        self.trainer.fit(X, Y, sample_weight=W, verbose=1)

        # self.features.append('2016_outcome')
        # sorted_features = np.argsort(self.trainer.feature_importances_)[::-1]
        # for idx in sorted_features[:10]:
        #     print(self.features[idx])
        # print(sorted_features)


    def predict(self):

        X = np.hstack((self.X_test, self.Y_test_16.reshape((len(self.Y_test_16), 1))))
        # X = self.X_test
        # X = self.Y_test_16.reshape((len(self.Y_test_16), 1))
        Y_pred = self.trainer.predict(X)
        Y_true = self.Y_test_20
        test_data = []
        for idx, entry in enumerate(self.test_set):
            print(entry[0],entry[1],round(Y_pred[idx],2),round(Y_true[idx],2))
        #     test_data.append({'location':entry,'prediction':Y_pred[idx],'actual':Y_true[idx]})
        # pprint.pprint(test_data)





elections = Elections()
# elections.download_2020_results()
# exit(0)
# elections.get_census_quickfacts('new york','wayne')
# elections.create_db()

# elections.get_census_all()

# elections.read_in_2016_results()
elections.load_data_from_db()
elections.separate_sets()
elections.set_trainer()

elections.cv()
elections.train()
elections.predict()