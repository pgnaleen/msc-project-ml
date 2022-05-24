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


access_logs = open('access.log', 'r')

# if __name__ == '__main__':


def model_train_predict(file_name):
    for row in access_logs:
        data_array = row.split(' ')
        url = data_array[6]
        code = data_array[8]

        features_list.append([url, code])

    df = pd.DataFrame(features_list, columns=['url', 'code'])

    print(df['url'].value_counts())
    print(df['code'].value_counts())
    # List of the columns we want to update from text to number
    columns = ['url', 'code']

    # Calling function for each column, one by one
    for col in columns:
        # Replace_data function takes two argument as described above.
        # 1 -  unique values on a column.
        # 2 -  Numbers list, so function can replace text with numbers
        # 3 -  Col is the column name, comes from the columns list
        replace_data(list(df[col].unique()), list(range(0, len(list(df[col].unique())))), col, df)

    print(df['url'].unique())
    print(df['code'].unique())

    # print the top 5 rows to see the changes
    print(df['url'].value_counts())
    print(df['code'].value_counts())

    d = df

    ################### Train Isolation Forest #################
    model = ensemble.IsolationForest(n_estimators=50, max_samples=50000, contamination=.003, max_features=2,
                                     bootstrap=False, n_jobs=1, random_state=1, verbose=0, warm_start=False)

    # train the model
    model.fit(d)

    # Predictions
    predictions = model.predict(d)
    print(predictions)

    # To Plot Predictions
    # plt.figure(figsize=(10, 6), dpi=150)
    # s = plt.scatter(d['url'], d['code'], c=predictions, cmap='coolwarm')
    # plt.colorbar(s, label='More Negative = More Anomalous')
    # plt.xlabel('Url', fontsize=16)
    # plt.ylabel('Code', fontsize=16)
    # plt.grid()
    # plt.title('Anomaly detection', weight='bold')
    # plt.show()

    log_file = open("C:\\xampp\\apache\\logs\\access.log", "r")
    # log_file = open(file_name, "r")
    log_lines = follow(log_file)
    # print(log_lines)
    for line in log_lines:
        print(line)
        # line_data_array = line.split(' ')
        # line_url = line_data_array[6]
        # line_code = line_data_array[8]
        # features_list.append([line_url, line_code])
        # line_data_frame = pd.DataFrame(features_list, columns=['url', 'code'])
        # print([line_url, line_code])

    #     # Calling function for each column, one by one
    #     for col in columns:
    #         # Replace_data function takes two argument as described above.
    #         # 1 -  unique values on a column.
    #         # 2 -  Numbers list, so function can replace text with numbers
    #         # 3 -  Col is the column name, comes from the columns list
    #         replace_data(list(line_data_frame[col].unique()), list(range(0, len(list(line_data_frame[col].unique())))), col, line_data_frame)
    #     print(line_data_frame.iloc[-1])
    #     print(model.predict(line_data_frame[len(line_data_frame) - 1:]))
    #     if model.predict(line_data_frame[len(line_data_frame) - 1:]) < -0.5:
    #         # send anomaly events to https://dashboard.pusher.com
    #         # anyone logged into this dashboard or via anomaly detection GUI this alerts can be seen
    #         pusher_client.trigger('my-channel', 'my-event',
    #                               {'message': 'Abnormal folder browsing found - '
    #                                           + line_url + ' at ' + line_data_array[3] + ' ' + line_data_array[4]})
