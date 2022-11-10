import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math
import itertools
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import prophet

"""# 2. Constants"""

DATA_PATH = 'data.csv'

TEST_SIZE = 100

"""# 3. Data Loading and Pre-processinng"""

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

data = {}
for location in set(df['location']):
    data[location] = df[df['location'] == location]

"""## 3.1 Calculate statistics"""

def get_stats(X):
    mean = sum(X) / len(X)
    sd = math.sqrt(sum([(xx - mean)*(xx - mean) for xx in X]) / (len(X)-1))
    return {'mean': mean, 'sd': sd, 'min': X.min(), 'max': X.max()}

def get_stats_all(data_):
    stats_ = {}
    for i in range(1, 7):
        attr = f'x{i}'
        stats_[attr] = get_stats(data_[attr])
    
    return stats_

stats = {}
for location in range(9):
    stats[location] = get_stats_all(data[location])

all_max = {}
all_min = {}
for i in range(6):
    attr = f'x{i+1}'
    all_max[attr] = max([v[attr]['max'] for v in stats.values()])
    all_min[attr] = min([v[attr]['min'] for v in stats.values()])

"""## 3.2 Filling in missing values"""

print("Checking for null values")
print(df.isnull().sum())

date_stats = {}
for location in range(9):
    d_ = data[location]['date']
    date_stats[location] = {'min': d_.min(), 'max': d_.max()}
print("The mininum and maximum dates are:")
print('\n'.join([f"{location}: {str(value)}" for location, value in date_stats.items()]))

MIN_DATE = pd.to_datetime('01-02-2015')
MAX_DATE = pd.to_datetime('01-31-2022')
date_range = pd.date_range(start=MIN_DATE, end=MAX_DATE)

def get_nearby_dt(date_, k_):
    return pd.date_range(start=(date_ - pd.Timedelta(days=k_)), end=(date_ + pd.Timedelta(days=k_)))

for location in range(9):
    # Calculate missing dates
    dates_considered = date_range.difference(data[location]['date'])
    d = data[location]

    # For each missing date
    for date in dates_considered:
        avg_values = []
        # Calculate dates in the nearby range
        k=2
        while len(avg_values) == 0:
            k += 1
            avg_dates = get_nearby_dt(date, k)
            avg_values = d[d['date'].isin(list(avg_dates))]

        # Obtain sum of values in surrounding range
        estimate = avg_values.sum(axis=0) / len(avg_values)
        estimate['date'] = date

        data[location] = data[location].append(estimate, ignore_index=True)

        if len(avg_dates) == 0:
            print(estimate)
    
    # Convert all attributes to their correct formats
    types = {f'x{i+1}': 'float64' for i in range(6)}
    data[location] = data[location].astype(types)

"""## 3.3 Excluding outliers for each location"""

def smoothen_outliers(X, mean, sd, sd_mult=3, win_size=7):
    wins_half = win_size // 2
    out = []
    for i, x in enumerate(X):
        if x < (mean - sd * sd_mult) or x > (mean + sd * sd_mult):
            smoothened_x = (X[i-wins_half:i] + X[i+1:i+wins_half+1]).sum() / (win_size-1)
            out.append(smoothened_x)
        else:
            out.append(x)
    return out

def smoothen_outliers_all(data_, stats_, sd_mult=3, win_size=7):
    for attr in stats_:
        data_[attr] = smoothen_outliers(data_[attr], stats_[attr]['mean'], stats_[attr]['sd'], sd_mult, win_size)
    return data_

for location in range(9):
    data[location] = smoothen_outliers_all(data[location], stats[location])

"""## 3.4 Normalization"""

def normalize(X, max_, min_):
    return (X - min_) / (max_ - min_)

for location in range(9):
    for i in range(6):
        attr = f'x{i+1}'
        data[location][attr] = normalize(data[location][attr], all_max[attr], all_min[attr])

"""# 4. Visualization"""

def plot_features_single_location(data_, stats=None, smoothen=True, outfile=None):
    fig, axs = plt.subplots(2,3, figsize=(16,10))
    fig.suptitle(f'Plots at location {location}')
    
    filtered_data = data_.copy()
    
    for i in range(6):
        a, b = i // 3, i % 3
        attr = f"x{i+1}"
        axs[a][b].plot(filtered_data[attr].tolist())
        axs[a][b].set_title(attr)
    
    
    if outfile:
        plt.savefig(outfile)
    
    plt.show()

plot_features_single_location(data[6])

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

"""# 5. Evaluation"""

def mae(x, y):
    return np.absolute(x-y).sum() / x.shape[0]

def comb(params_):
    return [dict(zip(params_.keys(), v)) for v in itertools.product(*params_.values())]

"""# 6. Facebook Prophet"""

def predict_prophet(params_, train_):
    # Generate predictions by parameter
    model = prophet.Prophet(**params_)
    model.fit(train_)
    future = model.make_future_dataframe(periods=100)
    pred = model.predict(future)[100]['yhat'].to_numpy()

    # Return values
    return pred

pr_data = data[6].copy()
pr_data['ds'] = pr_data['date']
pr_data['y'] = pr_data['x6']
pr_train, pr_test = train_test_split(pr_data, test_size=100)
y = pr_test['y'].to_numpy()

params = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

results = defaultdict(list)
plt.plot(y)
for params_ in comb(params):
    print(params_)
    pred = predict_prophet(params_, pr_train)
    for k, v in params_.items():
        results[k].append(v)
    
    # Calculate MAE
    results['mae'].append(mae(pred, y))

    # Plot results
    plt.plot(pred, label=str(params_))

print("Prophet results")
pd.DataFrame(results)

"""# 7. LSTM

## 7.1 Network
"""

class LSTMNet(nn.Module):
    def __init__(self, num_layers):
        super(LSTMNet, self).__init__()
        self.fc = nn.Linear(6, 12)
        self.lstm = nn.LSTM(12, 60, num_layers=num_layers)
        self.out = nn.Linear(60, 20)
    
    def forward(self, x):
        x = x
        x = torch.relu(self.fc(x))
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.out(x)
        return x

"""## 7.2 Data Transformation"""

def extract_features(data_):
    features = []
    targets = list(data_['x6'])
    for i in range(6):
        attr = f'x{i+1}'
        features.append(np.array(data_[attr]))
    features = np.stack(features).T

    return features, targets

STRIDE = 5

def generate_dataset(data_, stride):
    X = []
    y = []
    X_test = []
    y_test = []

    # Generate train dataset
    for location in range(9):
        features, targets = extract_features(data_[location])
        
        for i in range(0, len(features) - 420, stride):
            X.append(features[i: i + 400])
            y.append(targets[i + 400 : i + 420])
    
    # Generate test dataset from location 6
    features, targets = extract_features(data[6])
    
    for i in range(5):
        pred_index = len(features) - 20 * i - 20
        X_test.append(features[pred_index - 400 : pred_index])
        y_test.append(targets[pred_index : pred_index + 20])

    
    train_ds = TensorDataset(torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float64), torch.tensor(y_test, dtype=torch.float64))
    return train_ds, test_ds

train_ds, test_ds = generate_dataset(data, STRIDE)

"""## 7.3 Training loop"""

def train(params_, train_ds_):
    net = LSTMNet(params_['num_layers'])
    net.double()
    optimizer = params_['optimizer'](net.parameters(), lr=params_['lr'])
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_ds_, batch_size = params_['batch_size'])

    losses = []

    for epoch in range(3):
        for XX, yy in loader:
            optimizer.zero_grad()
            preds = net(XX)
            loss = loss_fn(preds, yy)
            loss.backward()
            optimizer.step()

            losses.append(loss.item() / XX.shape[0])
    
    return net, losses

def test(model, test_ds_):
    avg_mae = 0.0
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for XX, yy in test_ds_:
            XX = torch.unsqueeze(XX, 0)
            preds = net(XX).view(-1)
            preds_all.extend(list(preds))
            targets_all.extend(list(yy))
    avg_mae = mae(np.array(preds_all), np.array(targets_all))
    return avg_mae

"""## 7.5 Hyperparameter tuning and visualization"""

params_all = {
    'optimizer': [optim.Adam],
    'lr': [0.005, 0.01, 0.05],
    'batch_size': [8],
    'num_layers': [1, 2, 3]
}

results = defaultdict(list)
for params__ in comb(params_all):
    net, losses = train(params__, train_ds)
    for k, v in params__.items():
        results[k].append(v)
    
    avg_mae = test(net, test_ds)
    print(avg_mae)
    
    # Calculate MAE
    results['mae'].append(avg_mae)

print("RNN results")
print(pd.DataFrame(results))

params = {
    'changepoint_prior_scale': 0.01,
    'seasonality_prior_scale': 0.01,
}

pr_data = data[6].copy()
pr_data['ds'] = pr_data['date']
pr_data['y'] = pr_data['x6']

model = prophet.Prophet(**params)
model.fit(pr_data)
future = model.make_future_dataframe(periods=20)
pred = model.predict(future)['yhat'].to_numpy()

def denormalize(X, max_, min_):
    return (max_ - min_) * X + min_
np.savetxt('preds.csv', denormalize(pred[-20:], all_max['x6'], all_min['x6']), delimiter=',')

plt.show()