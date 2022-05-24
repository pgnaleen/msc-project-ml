# Create Artificial Data with Multivariate Outliers
from sklearn import ensemble
from utils import *
import time
import pusher

pusher_client = pusher.Pusher(
    app_id='1405172',
    key='d4bbdbf4283d76f108b6',
    secret='56aedf3bf31f73a135b5',
    cluster='ap2',
    ssl=True
)
features_list = []


def follow(the_file):
    the_file.seek(0, 2)
    while True:
        try:
            line = the_file.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line
        except PermissionError:
            print("permission error")


# open and read the file after the appending:
f = open("datasetLocation.txt", "r")

# access_logs = open('access_60K.log', 'r')
access_logs = open(f.read(), 'r')  # here opens file inside datasetLocation.txt

for row in access_logs:
    line_data_array = row.split(' ')
    line_url = line_data_array[6]
    line_code = line_data_array[8]
    features_list.append([line_url, line_code])

training_data_frame = pd.DataFrame(features_list, columns=['url', 'code'])

# List of the columns we want to update from text to number
columns = ['url', 'code']
dfvalue = training_data_frame['url'].value_counts()

# why? - otherwise training data frame will have class as well and model consider class as feature
training_data_frame_class = training_data_frame.copy()
for index, row in training_data_frame.iterrows():
    training_data_frame_class.at[index, 'class'] = 1 if dfvalue[row['url']] > 1500 else 0

# Calling function for each column, one by one
for col in columns:
    # Replace_data function takes two argument as described above.
    # 1 -  unique values on a column.
    # 2 -  Numbers list, so function can replace text with numbers
    # 3 -  Col is the column name, comes from the columns list
    replace_data(list(training_data_frame[col].unique()), list(range(0, len(list(training_data_frame[col].unique())))),
                 col, training_data_frame)

print("unique urls")
print(training_data_frame['url'].unique())
print(training_data_frame['code'].unique())

# print the top 5 rows to see the changes
print(training_data_frame.head())
print(training_data_frame['url'].value_counts())
print(training_data_frame['code'].value_counts())

################### Train Isolation Forest #################
model = ensemble.IsolationForest(n_estimators=50, max_samples=50000, contamination=.003, max_features=2,
                                 bootstrap=False, n_jobs=1, random_state=1, verbose=0, warm_start=False)

# train the model
model.fit(training_data_frame)

predictions = model.predict(training_data_frame)
training_data_frame['predicted_class'] = predictions
training_data_frame['class'] = training_data_frame_class['class']

for index, row in training_data_frame.iterrows():
    training_data_frame.at[index, 'predicted_class'] = 1 if training_data_frame.at[
                                                                index, 'predicted_class'] > 0.5 else 0

training_data_frame = training_data_frame.astype(int)

# To Plot Predictions
plt.figure(figsize=(10, 6), dpi=150)
s = plt.scatter(training_data_frame['url'], training_data_frame['code']
                , c=training_data_frame['predicted_class'], cmap='coolwarm')
plt.colorbar(s, label='More Negative = More Anomalous')
plt.xlabel('Url', fontsize=16)
plt.ylabel('Code', fontsize=16)
plt.grid()
plt.title('Anomaly detection', weight='bold', fontsize=20)
plt.show()

# print the result, just pass the prediction and test data for both functions.
print(confusion_matrix(training_data_frame['class'], training_data_frame['predicted_class']))
print(classification_report(training_data_frame['class'], training_data_frame['predicted_class']))

if __name__ == '__main__':
    # open and read the file after the appending:
    f = open("accessLogLocation.txt", "r")

    log_file = open(f.read(), 'r')  # here opens file inside accessLogLocation.txt
    # log_file = open("C:\\xampp\\apache\\logs\\access.log", "r")

    log_lines = follow(log_file)
    for line in log_lines:
        print(line)
        line_data_array = line.split(' ')
        line_url = line_data_array[6]
        line_code = line_data_array[8]
        features_list.append([line_url, line_code])
        training_data_frame = pd.DataFrame(features_list, columns=['url', 'code'])

        # Calling function for each column, one by one
        for col in columns:
            # Replace_data function takes two argument as described above.
            # 1 -  unique values on a column.
            # 2 -  Numbers list, so function can replace text with numbers
            # 3 -  Col is the column name, comes from the columns list
            replace_data(list(training_data_frame[col].unique()),
                         list(range(0, len(list(training_data_frame[col].unique())))), col, training_data_frame)
        print(training_data_frame.iloc[-1])
        print(model.predict(training_data_frame[len(training_data_frame) - 1:]))
        if model.predict(training_data_frame[len(training_data_frame) - 1:]) < -0.5:
            # send anomaly events to https://dashboard.pusher.com
            # anyone logged into this dashboard or via anomaly detection GUI this alerts can be seen
            pusher_client.trigger('my-channel', 'my-event',
                                  {'message': 'Abnormal folder browsing found - '
                                              + line_url + ' at ' + line_data_array[3] + ' ' + line_data_array[4]})
