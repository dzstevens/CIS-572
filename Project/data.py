import csv, random
import numpy as np
from collections import defaultdict
from datetime import datetime


def get_data(file_):
    with open(file_, newline='') as f:
        r = csv.DictReader(f)
        d = defaultdict(list)
        for row in r:
            if row['Status'] in neg | pos:
                for k, v in row.items():
                    if k not in ignores:
                        if k in missing_values:
                            d[k+' Missing'].append(0 if v else 1)
                        if k in percents:
                            d[k].append(round(float(v[:-1])*100) if v else 0)
                        elif k in ints:
                            d[k].append(int(v) if v else 0)
                        elif k in bools:
                            d[k].append(0 if v == bools[k] else 1)
                        elif k in floats:
                            d[k].append(round(float(v)*100) if v else 0)
                        elif k == 'Status':
                            d[k].append(0 if v in neg else 1)
                        elif k == 'State':
                            d[k+' Population'].append(pops[v])
                            for reg, states in regions.items():
                                d['Region: '+reg].append(1 if v in states else 0)
                        elif k in multis:
                            for val in multis[k]:
                                key = k + ': '
                                key +=  val if val else 'Missing'
                                d[key].append(1 if v == val else 0)
                        elif k == 'Earliest CREDIT Line':
                            d['CREDIT History Length (Days)'].append(
                                (date(row['Application Date']) -
                                    date(v)).days)
                        elif k == 'Issued Date':
                            d['Application Days Pending'].append(
                                (date(v) -
                                    date(row['Application Date'])).days)
                        elif k == 'Application Expiration Date':
                            d['Application Days Given'].append(
                                (date(v) -
                                    date(row['Application Date'])).days)
                        elif k == 'Application Date':
                            for i in range(len(months)):
                                d['Application Month: '+months[i]].append(
                                    1 if date(v).month == i else 0)
                        else:
                            raise KeyError(k)
        return d

def get_test_data(data, test_perc=0.3, seed=42):
    random.seed(seed)
    l = len(data[sorted(data).pop()])
    test_data = defaultdict(list)
    for i in sorted(random.sample(range(l), round(l*test_perc)), reverse=True):
        for k, v in data.items():
            test_data[k].append(v.pop(i))
    return test_data

def get_target(data, feat='Status'):
    return data.pop(feat)

 
def date(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d') if date_str else None

def dump_data(data, f_name, t_name='Status'):
    keys = sorted(data)
    keys.remove(t_name)
    keys.append(t_name)
    l = len(data[keys[0]])
    with open(f_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows({k:v[i] for k, v in data.items()} for i in range(l))

def load_data(f_name):
    with open(f_name, 'r', newline='') as f:
        reader = csv.DictReader(f)
        d = defaultdict(list)
        for row in reader:
            for k, v in row.items():
                try:
                    d[k].append(int(v))
                except ValueError:
                    print(k,v)
                    raise
    return d

def numpit(data, target='Status'):
    T = np.array(data.pop(target))
    h = sorted(data)
    X = np.array([row for row in [[data[k][i] for k in h] for i in range(len(T))]])
    return h, X, T

def load(f_name, t_name='Status'):
    return numpit(load_data(f_name), target=t_name)

def denump(h, X, T, t_name='Status'):
    d = defaultdict(list)
    for i in range(len(X)):
        row = X[i]
        for j in range(len(h)):
            d[h[j]].append(row[j])
        d[t_name].append(T[i])
    return d

def dump(f_name, h, X, T, t_name='Status'):
    dump_data(denump(h, X, T, t_name), f_name, t_name)

ignores = set(['Loan ID', 'Amount Funded By Investors', 'Loan Description',
               'Total Amount Funded', 'Remaining Principal Funded by Investors',
               'Payments To Date (Funded by investors)', 'Remaining Principal ',
               ' Payments To Date', 'City', 'Screen Name', 'Code', 'Loan Title',
               'Accounts Now Delinquent', 'Delinquent Amount', 'State'])

percents = set(['Debt-To-Income Ratio', 'Interest Rate', 'Revolving Line Utilization'])

dates = set(['Application Date', 'Application Expiration Date', 'Issued Date', 'Earliest CREDIT Line'])

ints = set(['Accounts Now Delinquent', 'Delinquencies (Last 2 yrs)', 'Inquiries in the Last 6 Months',
            'Months Since Last Delinquency', 'Months Since Last Record', 'Open CREDIT Lines',
            'Public Records On File', 'Revolving CREDIT Balance', 'Total CREDIT Lines', 
            'Open CREDIT Lines', 'Delinquent Amount'])

missing_values = set(['Months Since Last Record', 'Months Since Last Delinquency',
                     'Revolving Line Utilization'])

bools = {'Education':'', 'Initial Listing Status':'F', 'Loan Length': '36 months'}

floats = set(['Amount Requested', 'Monthly Income', 'Monthly PAYMENT'])

neg = set(['Charged Off', 'Default'])
pos = set(['Fully Paid'])

pops = {'AK': 731449, 'AL': 4822023, 'AR': 2949131, 'AZ': 6553255, 'CA': 38041430,
        'CO': 5187582, 'CT': 3590347, 'DC': 632323, 'DE': 917092, 'FL': 19317568,
        'GA': 9919945, 'HI': 1392313, 'IA': 6537334, 'ID': 1595728, 'IL': 12875255,
        'IN': 6537334, 'KS': 2885905, 'KY': 4380415, 'LA': 4601893, 'MA': 6646144,
        'MD': 5884563, 'ME': 1329192, 'MI': 9883360, 'MN': 5379139, 'MO': 6021988,
        'MS': 2984926, 'MT': 1005141, 'NC': 9752073, 'ND': 699628, 'NE': 1855525,
        'NH': 1320718, 'NJ': 8864590, 'NM': 2085538, 'NV': 2758931, 'NY': 19570261,
        'OH': 11544225, 'OK': 3814820, 'OR': 3899353, 'PA': 12763536, 'RI': 1050292,
        'SC': 4723723, 'SD': 833354, 'TN': 6456243, 'TX': 26059203, 'UT': 2855287,
        'VA': 8185867, 'VT': 626011, 'WA': 6897012, 'WI': 5726398, 'WV': 1855413,
        'WY': 576412}

regions = {'Pacific': ('AK', 'WA', 'OR', 'CA', 'HI'),
           'Mountain': ('MT', 'ID', 'WY', 'NV', 'UT', 'CO'),
           'Southwest': ('AZ', 'NM', 'OK', 'TX'),
           'Tornados': ('ND', 'SD', 'NE', 'KS', 'IA', 'MO'),
           'Lakes': ('MN', 'WI', 'IL', 'IN', 'MI', 'OH'),
           'South': ('AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'TN',
                     'SC', 'NC', 'KY', 'VA', 'WV'),
           'S. New England': ('MD', 'DE', 'PA', 'NJ', 'NY'),
           'N. New England': ('CT', 'RI', 'MA', 'VT', 'NH', 'ME')}

credit_grades = [l+i for l in 'ABCDEFG' for i in '12345']

months = [datetime.strftime(date('2000-{:}-01'.format(x)), '%b')
          for x in range(1,13)]

emp_length = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
              '5 years', '6 years', '7 years', '8 years', '9 years',
              '10+ years', 'n/a']

home_owner = ['MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT']

fico_range = ['', '660-664', '665-669', '670-674', '675-679', '680-684',
              '685-689', '690-694', '695-699', '700-704', '705-709', '710-714',
              '715-719', '720-724', '725-729', '730-734', '735-739', '740-744',
              '745-749', '750-754', '755-759', '760-764', '765-769', '770-774',
              '775-779', '780-784', '785-789', '790-794', '795-799', '800-804',
              '805-809', '810-814', '815-819', '820-824', '825-829', '830-834',
              '835-839', '840-844']

loan_purpose = ['car', 'credit_card', 'debt_consolidation', 'educational',
                'home_improvement', 'house', 'major_purchase', 'medical',
                'moving', 'other', 'renewable_energy', 'small_business',
                'vacation', 'wedding']

months = [datetime.strftime(datetime.strptime(str(x), '%m'), '%b')
            for x in range(1,13)]

multis = {'CREDIT Grade': credit_grades, 'Employment Length': emp_length,
          'Employment Length': emp_length, 'Home Ownership': home_owner,
          'FICO Range': fico_range, 'Loan Purpose': loan_purpose}

