
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from logging import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
!pip install kneed
from kneed import KneeLocator, DataGenerator as dg

def comb_file(input_file1, input_file2):
  df1 = pd.read_excel(input_file1)
  df2 = pd.read_excel(input_file2)
  # df1 = pd.read_csv(input_file1)
  # df2 = pd.read_csv(input_file2)
  df1.drop(['B_M','Inputs'],axis=1,inplace=True)
  C = []
  for i in range(len(df1)):
    C.append(0)
  df1['B_M'] = C
  print
  df2.drop(['B_M','Inputs'],axis=1,inplace=True)
  U = []
  for i in range(len(df2)):
    U.append(1)
  df2['B_M'] = U

  df3 = pd.concat([df1,df2],ignore_index=True)
  input = []
  for i in range(len(df3)):
    input.append(i+1)
  idx = 0
  df3.insert(loc=idx, column='Inputs', value=input)
  return df3

df3 = comb_file('/content/M_B_con_200.xlsx','/content/M_B_uncon_200.xlsx')

# df3 = comb_file('/content/self_attn_con.csv','/content/self_attn_uncon.csv')

df3.info()

seed = 30

from sklearn.ensemble import RandomForestClassifier
def logistic_regression_skbst(input_file, k_values):
    try:
        df = pd.read_excel(input_file)
    except:
        df = input_file
    df['B_M'] = df.B_M

    X = df.drop(['B_M','Inputs'], axis=1)
    y = df['B_M']

    X_df = pd.DataFrame(X, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # List to store accuracy scores for different k values
    accuracy_scores = []

    for k in k_values:
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features = selector.get_support(indices=True)

        # Print selected features
        selected_column_names = X_train.columns[selected_features]
        # print(f"Selected features for k={k} (indices): {selected_features}")
        # print(f"Selected feature names for k={k}: {selected_column_names.tolist()}")

        # Train logistic regression model
        model = RandomForestClassifier(random_state=1)
        # model = LogisticRegression(max_iter=200)
        model.fit(X_train_selected, y_train)

        # Test the model
        X_test_selected = selector.transform(X_test)
        y_pred = model.predict(X_test_selected)

        # Calculate accuracy and store it
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"\nAccuracy for k={k}: {accuracy}")
        # print(f"Confusion Matrix for k={k}:\n", conf_matrix)
        # print(f"Classification Report for k={k}:\n", class_report)
    kl = KneeLocator(k_values, accuracy_scores, curve="concave")
    kl.plot_knee()
    # Plotting accuracy vs k for elbow plot
    plt.figure(figsize=(15, 8))
    plt.plot(k_values, accuracy_scores, marker='o', color='b')
    plt.title('Elbow Plot: Accuracy vs Number of Features (k)', fontsize=14)
    plt.xlabel('Number of Features (k)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    best_k_max = k_values[np.argmax(accuracy_scores)]
    print(f"\nBest k based on maximum accuracy: {best_k_max}")

    # 2. Use rate of change to detect plateau (acceleration of accuracy)
    accuracy_differences = np.diff(accuracy_scores)
    best_k_rate_of_change = k_values[np.argmax(accuracy_differences < 0.01)]  # Threshold for plateau
    print(f"Best k based on rate of change: {best_k_rate_of_change}")

    # 3. Find the k with the lowest variance and highest accuracy
    variance = np.var(accuracy_scores)
    best_k_variance = k_values[np.argmin(accuracy_scores)]
    print(f"Best k based on stability (low variance): {best_k_variance}")

    # Return the best k values
    return model, X, y, accuracy_scores, best_k_max, best_k_rate_of_change, best_k_variance

k_values = list(range(1, len(df3.columns)-1))  # Example k values to test, change as needed
logistic_regression_skbst(df3, k_values)

# for category_value in df3['Category'].unique():
#   print(category_value)
#   logistic_regression_skbst(df3[df3['Category'] == category_value], k_values)

def logistic_regression_skbst_grouped(input_file, k_values, accuracy_threshold=0.001):
    try:
        df = pd.read_excel(input_file)
    except:
        df = input_file
    df['B_M'] = df.B_M

    X = df.drop(['B_M', 'Inputs'], axis=1)
    y = df['B_M']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    accuracy_scores = []
    k_accuracy_dict = {}

    # Try different values of k for SelectKBest
    for k in k_values:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Train the model and evaluate accuracy
        model = RandomForestClassifier(random_state=1)
        # model = LogisticRegression(max_iter=200)
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        accuracy_scores.append(accuracy)
        k_accuracy_dict[k] = accuracy

    # Group k values by similar accuracy
    grouped_k = {}
    for k, acc in k_accuracy_dict.items():
        # Find group based on accuracy range (e.g., within a threshold of 0.01)
        grouped = False
        for group_acc in grouped_k.keys():
            if abs(group_acc - acc) < accuracy_threshold:
                grouped_k[group_acc].append(k)
                grouped = True
                break
        if not grouped:
            grouped_k[acc] = [k]

    # Sort the groups by accuracy (descending)
    sorted_groups = sorted(grouped_k.items(), key=lambda item: item[0], reverse=True)

    # Plotting the grouped data
    plt.figure(figsize=(8, 6))
    for accuracy, k_group in sorted_groups:
        plt.scatter(k_group, [accuracy] * len(k_group), label=f'Accuracy: {accuracy:.2f}')

    plt.title('Grouped Accuracy vs Number of Features (k)', fontsize=14)
    plt.xlabel('Number of Features (k)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(title="Accuracy groups")
    plt.grid(True)
    plt.show()

    return k_accuracy_dict, grouped_k
k_values = list(range(1, len(df3.columns)-1))  # Example k values to test, change as needed
k_accuracy_dict, grouped_k = logistic_regression_skbst_grouped(df3, k_values)
min_values = [(key, min(value_list)) for key, value_list in grouped_k.items()]
x_key = []
y_value = []
# Print the result
for key, min_value in min_values:
  x_key.append(key)
  y_value.append(min_value)

kl = KneeLocator(y_value, x_key, curve="concave")
kl.plot_knee(figsize=(8, 3))


# plt.xticks(np.arange(0, (y_value[-1])+1, step=10))

# plt.xlim(0,10)
# xticks = np.array([0, 10, 19, 30, 40, 50, 60])
# plt.xticks(xticks)

xticks = np.arange(0, (y_value[-1]) + 1, step=10)
xticks = np.append(xticks, 1)  # Add the value 3 to the xticks
plt.xticks(np.unique(xticks))  # Use np.unique to ensure no duplicates

plt.xlabel("Number of Features (k)",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.title("Elbow Method",fontsize=20)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.legend(fontsize=15)
plt.savefig('/content/mist7be.svg', format='svg')
plt.show()

from sklearn.preprocessing import MinMaxScaler
def logistic_regression_skbst(input_file,k):
  try :
    df = pd.read_excel(input_file)
  except :
    df = input_file
  df['B_M'] = df.B_M

  X = df.drop(['B_M','Inputs'], axis=1)
  y = df['B_M']

  # # Step 3: Initialize the MinMaxScaler
  # scaler = MinMaxScaler()

  # # Step 4: Fit and transform the feature columns
  # X_scaled = scaler.fit_transform(X)

  X_df = pd.DataFrame(X, columns=X.columns)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

  selector = SelectKBest(score_func=f_classif, k=k)

  X_train_selected = selector.fit_transform(X_train, y_train)
  selected_features = selector.get_support(indices=True)
  print(selector)
  print(f"Selected features (indices): {selected_features}")


  selected_column_names = X_train.columns[selected_features]
  print(f"Selected feature names: {selected_column_names.tolist()}")

  # model = LogisticRegression(max_iter=200)
  model = RandomForestClassifier(random_state=1)


  model.fit(X_train_selected, y_train)

  X_test_selected = selector.transform(X_test)
  y_pred = model.predict(X_test_selected)

  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  class_report = classification_report(y_test, y_pred)
  conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

  print("Accuracy:", accuracy)
  print("\nConfusion Matrix:")
  print(conf_matrix)
  print("\nClassification Report:")
  print(class_report)

  plt.figure(figsize=(6, 5))
  ax = sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', annot_kws={"size": 25},cbar=False)
  # for i in range(conf_matrix_percentage.shape[0]):
  #   for j in range(conf_matrix_percentage.shape[1]):
  #       ax.text(j+0.8, i+0.5, '%', ha='center', va='center', fontsize=20, color='black')
  plt.title(f'Overall Accuracy: {accuracy:.2%}',fontsize=25)
  plt.xlabel('Predicted',fontsize=25)
  plt.ylabel('True',fontsize=25)
  ax.tick_params(axis='x', labelsize=20)  # Set font size for x-axis ticks
  ax.tick_params(axis='y', labelsize=20)
  # colorbar = ax.collections[0].colorbar
  # colorbar.ax.tick_params(labelsize=20)
  plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
  plt.savefig('/content/mist7bc.svg', format='svg')

  plt.show()
  return model, X, y, accuracy, plt

model, X, y, accuracy, plt = logistic_regression_skbst(df3,43)

!pip install kaleido

df = pd.read_csv('/content/judge_seed42.csv',encoding='latin')

def extract_values_from_string(cell_value):
    matches = re.findall(r'\((\w)\)\s*(\d)', cell_value)
    # matches = re.findall(r'\((\w)\)(\d)', cell_value)

    print(matches)

    data_dict = {label: int(value) for label, value in matches}

    return data_dict

judge_columns = [col for col in df.columns if 'judge' in col.lower()]

for col in judge_columns:
    print(col)
    extracted_data = df[col].apply(extract_values_from_string)
    extracted_df = pd.DataFrame(extracted_data.tolist(), index=df.index)
    extracted_df.columns = [f'{col}_{label}' for label in extracted_df.columns]
    # extracted_df = extracted_df.apply(lambda x: x.apply(lambda val: 0 if pd.isna(val) or val <= 2 else (1 if val >= 5  else 1)))
    # extracted_df = extracted_df.apply(lambda x: x.apply(lambda val: 1 if val == 5 else 0))

    extracted_df = extracted_df.apply(lambda x: x.apply(lambda val: 1 if pd.isna(val) else (1 if val > 1 else val)))

    df = pd.concat([df, extracted_df], axis=1)



print("\nUpdated DataFrame:")
# df

df.to_csv('/content/malicious_model_q2judgenew_llama1b_updated.csv', index=False)

df = pd.read_csv('/content/malicious_model_q2judgenew_llama1b_updated.csv',encoding='latin-1')
df.columns = [col.replace('base_', 'response_') if col.startswith('base_') else col for col in df.columns]

# # df = pd.read_csv('/content/malicious_model_q2judgenew_llama1b_updated.csv',encoding='latin-1')
# df = df.loc[:, ['Category','response_q_Judge_a' ,'response_qs_Judge_a','response_q_Judge_b' ,'response_qs_Judge_b','response_q_Judge_c' ,'response_qs_Judge_c']]
# df['Max_Value_a'] = df[['response_q_Judge_a' ,'response_qs_Judge_a']].max(axis=1)
# df['Max_Value_b'] = df[['response_q_Judge_b' ,'response_qs_Judge_b']].max(axis=1)
# df['Max_Value_c'] = df[['response_q_Judge_c' ,'response_qs_Judge_c']].max(axis=1)
# # df_selected = df[['Category', 'response_model_Judge_b','Max_Value']]
# # df['Max_Value'] = df[['response_model_Judge_b','response_model_Judge_a','response_model_Judge_c']].max(axis=1)

# df_selected = df[['Category','response_q_Judge_b' ,'response_qs_Judge_b','Max_Value_b']]
# # df_selected = df[['Category', 'response_model_Judge_b','Max_Value']]
# df_aggregated = df_selected[['Category', 'response_q_Judge_b' ,'response_qs_Judge_b']].groupby('Category', as_index=False).mean()

# df = pd.read_csv('/content/malicious_model_q2judgenew_llama1b_updated.csv',encoding='latin-1')
df = df.loc[:,['Category', 'response_p_Judge_a','response_ps_Judge_a','response_p_Judge_b','response_ps_Judge_b','response_p_Judge_c','response_ps_Judge_c']]
df['Max_Value_a'] = df[['response_p_Judge_a','response_ps_Judge_a']].max(axis=1)
df['Max_Value_b'] = df[['response_p_Judge_b','response_ps_Judge_b']].max(axis=1)
df['Max_Value_c'] = df[['response_p_Judge_c','response_ps_Judge_c']].max(axis=1)
# df['Max_Value'] = df[['response_model_Judge_b','response_model_Judge_a','response_model_Judge_c']].max(axis=1)
# df_selected = df[['Category', 'response_model_Judge_b','Max_Value']]

df_selected = df[['Category', 'response_p_Judge_b','response_ps_Judge_b','Max_Value_b']]
# # # df_selected = df[['Category', 'response_model_Judge_b','Max_Value']]
df_aggregated = df_selected[['Category', 'response_p_Judge_b','response_ps_Judge_b']].groupby('Category', as_index=False).mean()


# # df = pd.read_csv('/content/malicious_model_q2judgenew_llama1b_updated.csv',encoding='latin-1')
# # df = df[['Category', 'response_p_Judge_a','response_ps_Judge_a','response_p_Judge_b','response_ps_Judge_b','response_p_Judge_c','response_ps_Judge_c']]
# # df['Max_Value_a'] = df[['response_p_Judge_a','response_ps_Judge_a']].max(axis=1)
# # df['Max_Value_b'] = df[['response_p_Judge_b','response_ps_Judge_b']].max(axis=1)
# # df['Max_Value_c'] = df[['response_p_Judge_c','response_ps_Judge_c']].max(axis=1)
# df['Max_Value'] = df[['response_model_Judge_b','response_model_Judge_a','response_model_Judge_c']].max(axis=1)
# df['Max_Value_bp'] = df[['response_p_Judge_b','response_ps_Judge_b']].max(axis=1)
# df['Max_Value_bq'] = df[['response_q_Judge_b' ,'response_qs_Judge_b']].max(axis=1)
# df_selected = df[['Category', 'response_model_Judge_b','Max_Value','Max_Value_bp','Max_Value_bq']]

# # df_selected = df[['Category', 'response_p_Judge_b','response_ps_Judge_b','Max_Value_b']]
# # df_selected = df[['Category', 'response_model_Judge_b','Max_Value']]
# df_aggregated = df_selected[['Category', 'response_model_Judge_b','Max_Value_bp','Max_Value_bq']].groupby('Category', as_index=False).mean()

a_columns = [col for col in df.columns if '_a' in col]
b_columns = [col for col in df.columns if '_b' in col]
c_columns = [col for col in df.columns if '_c' in col]

def find_max_b_a_c_for_noise(row):
    max_b_values = []
    max_a_values = []
    max_c_values = []

    # ns = ['q', 'qs']
    ns = ['p', 'ps']
    for noise_level in ns:
        # a_col = f'response_model_noise_{noise_level}_changed_Judge_a'
        # b_col = f'response_model_noise_{noise_level}_changed_Judge_b'
        # c_col = f'response_model_noise_{noise_level}_changed_Judge_c'

        a_col = f'response_{noise_level}_Judge_a'
        b_col = f'response_{noise_level}_Judge_b'
        c_col = f'response_{noise_level}_Judge_c'
        b_value = row[b_col]
        a_value = row[a_col]
        c_value = row[c_col]

        max_b_values.append(b_value)
        max_a_values.append(a_value)
        max_c_values.append(c_value)

    # a_col = f'response_model_Judge_a'
    # b_col = f'response_model_Judge_b'
    # c_col = f'response_model_Judge_c'

    b_value = row[b_col]
    a_value = row[a_col]
    c_value = row[c_col]

    max_b_values.append(b_value)
    max_a_values.append(a_value)
    max_c_values.append(c_value)

    max_b_value = max(max_b_values)

    max_b_indices = [i for i, b in enumerate(max_b_values) if b == max_b_value]

    if len(max_b_indices) > 1:
        corresponding_a_values = [max_a_values[i] for i in max_b_indices]
        max_a_value = max(corresponding_a_values)

        max_a_index = corresponding_a_values.index(max_a_value)
        max_c_value = max_c_values[max_b_indices[max_a_index]]

        return pd.Series([max_b_value, max_a_value, max_c_value])

    max_b_index = max_b_indices[0]
    return pd.Series([max_b_values[max_b_index], max_a_values[max_b_index], max_c_values[max_b_index]])

df[['Max_b_Value', 'Max_a_Value', 'Max_c_Value']] = df.apply(find_max_b_a_c_for_noise, axis=1)

df_aggregated

mean_values =pd.concat([
    df.filter(like='Judge_a').apply(pd.to_numeric, errors='coerce'),
    df.filter(like='Judge_b').apply(pd.to_numeric, errors='coerce'),
    df.filter(like='Judge_c').apply(pd.to_numeric, errors='coerce'),
    df.filter(like='Judge_d').apply(pd.to_numeric, errors='coerce'),
    df.filter(like='Judge_e').apply(pd.to_numeric, errors='coerce'),
    df.filter(like='Max_a_Value').apply(pd.to_numeric, errors='coerce'),
    df.filter(like='Max_b_Value').apply(pd.to_numeric, errors='coerce'),
    df.filter(like='Max_c_Value').apply(pd.to_numeric, errors='coerce')
]).mean()
mean_values

!pip install --upgrade plotly

!plotly_get_chrome

import plotly.graph_objects as go
import pandas as pd
from kaleido import Kaleido
category_mapping = {
    'Disinformation': 'D',
    'Economic harm': 'EH',
    'Expert advice': 'EA',
    'Fraud/Deception': 'FD',
    'Government decision-making': 'GDM',
    'Harassment/Discrimination': 'HD',
    'Malware/Hacking': 'MH',
    'Physical harm': 'PH',
    'Privacy': 'P',
    'Sexual/Adult content': 'SAC'
}
# Reshape the dataframe for plotly
df_melted = pd.melt(df_aggregated, id_vars=['Category'], var_name='Model', value_name='Value')
df_melted['Category'] = df_melted['Category'].map(category_mapping)
# Increase all values by 1.5
df_melted['Value'] = df_melted['Value']

# Create the radar plot
fig = go.Figure()

# Function to ensure closing the loop

def close_loop(df, model_name):
    data = df[df['Model'] == model_name]
    r_values = data['Value'].tolist()
    theta_values = data['Category'].tolist()

    # Append first value at the end to close the shape
    if len(r_values) > 0:
        r_values.append(r_values[0])
        theta_values.append(theta_values[0])

    return r_values, theta_values

# Define models with their corresponding legend names and colors
models = {
    'response_model_Judge_b': ('Original Model', 'green'),
    # 'response_model_noise_0.1_changed_Judge_b': ('Noise 0.1', 'blue'),
    # 'response_model_noise_0.2_changed_Judge_b': ('Noise 0.2', 'orange'),
    # 'response_model_noise_0.3_changed_Judge_b': ('Noise 0.3', 'red'),
    'Max_Value_bp': ('Avg Best', 'red')
}

# Add traces with specified colors
for model, (legend_name, color) in models.items():
    r, theta = close_loop(df_melted, model)
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        name=legend_name,
        line=dict(color=color)  # Set custom color
    ))

# Update layout for better visualization
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickfont=dict(size=15)# Adjusted range after increasing values
        ),
        angularaxis=dict(
            tickfont=dict(
                size=30,  # Increase the font size for category labels
                color='black'  # Optional: Change the color of the labels
            )
        )
    ),
    showlegend=True,
    legend=dict(
        font=dict(
            size=15,  # Increase the legend font size
            color='black'  # Optional: Change the color of the legend text
        ),
        title=dict(
            font=dict(
                size=1  # Optional: Increase the legend title font size
            )
        ),
        x=0.8,  # Position the legend horizontally
        y=1.1  # Position the legend vertically
    ),
    width=600,  # Increased width
    height=400,  # Increased height
        margin=dict(
        l=0,  # Left margin
        r=0,  # Right margin
        t=0,  # Top margin
        b=30   # Bottom margin
    )

)

fig.write_image("/content/radio14.svg")

fig.show()

df_con  = pd.read_excel('/content/M_B_con_200.xlsx',nrows=100)
activation_columns = df_con.filter(like='Activation_31', axis=1)
# activation_columns = df.filter(like='response_model_Judge_b', axis=1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

np.random.seed(42)
data = activation_columns.copy()
data['category'] = df_con['Category']

# data['category'] = df3['Category']


scaler = StandardScaler()
scaled_data =  scaler.fit_transform(activation_columns.copy())

inertia = []
sil_scores = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

    if k > 1:
        sil_scores.append(silhouette_score(scaled_data, kmeans.labels_))

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', label='Inertia (SSE)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()




plt.figure(figsize=(8, 5))
plt.plot(k_range[1:], sil_scores, marker='o', color='orange', label='Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.grid(True)
plt.show()

optimal_k = 2

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_data)

sil_score = silhouette_score(scaled_data, data['cluster'])
print(f"Silhouette Score: {sil_score:.3f}")

# pca = PCA(n_components=2)
# pca_data = pca.fit_transform(scaled_data)

# plt.figure(figsize=(8, 6))
# plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['cluster'], cmap='viridis', alpha=0.6, s=50)

# pca_centers = pca.transform(kmeans.cluster_centers_)
# plt.scatter(pca_centers[:, 0], pca_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')

# for i, center in enumerate(pca_centers):
#     plt.annotate(f'Cluster {i+1}', (center[0], center[1]), fontsize=12, color='black',
#                  xytext=(5, 5), textcoords='offset points', weight='bold')

plt.title(f'KMeans Clustering with {optimal_k} clusters (PCA-Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

print("Cluster assignments for each data point:")
print(data[['category', 'cluster']])

# print(data[['category', 'cluster']].value_counts())
data['cluster'].unique()

# Step 8: Calculate and print the category distribution and percentage for each cluster
cluster_info = data.groupby(['cluster', 'category']).size().unstack(fill_value=0)
cluster_info_percentage = cluster_info.div(cluster_info.sum(axis=1), axis=0) * 100

# Display the category distribution
print("\nCategory Distribution in Each Cluster:")
print(cluster_info.T)

max_cluster_per_category = cluster_info_percentage.idxmax(axis=0)
max_percentage_per_category = cluster_info_percentage.max(axis=0)

print("\nCategory and Max Cluster it Belongs To (with percentage):")
for category, cluster in max_cluster_per_category.items():
    percentage = max_percentage_per_category[category]
    print(f"Category: {category}, Max Cluster: {cluster}, Percentage: {percentage:.2f}%")

data

category_distribution = pd.crosstab(data['cluster'], data['category'], normalize='index') * 100

# Display the distribution as percentages|
print(category_distribution.T)

import pandas as pd

# Assuming df is your DataFrame
# Get the count of samples per category
category_counts = data['category'].value_counts()

# Now, calculate the distribution of clusters within each category
category_cluster_distribution = pd.crosstab(data['category'], data['cluster'], normalize='index') * 100

category_cluster_distribution

data1 = pd.read_csv('/content/malicious_model_q2judgenew_llama1b_updated.csv',encoding='latin-1')
data[['response_q_Judge_b','response_qs_Judge_b','response_p_Judge_b','response_ps_Judge_b']] = data1[['response_q_Judge_b','response_qs_Judge_b','response_p_Judge_b','response_ps_Judge_b']]
data

data_aggregated = data[['cluster','response_q_Judge_b','response_qs_Judge_b','response_p_Judge_b','response_ps_Judge_b']].groupby('cluster', as_index=False).mean()
data_aggregated

data[['cluster','response_p_Judge_b']].value_counts()

# Calculate silhouette scores for each category separately
categories = data['category'].unique()
silhouette_scores = {}

for category in categories:
    # Filter data for this category
    category_data = data[data['category'] == category]

    # Calculate silhouette score for this category using only the "Value" column
    score = silhouette_score(category_data[['response_model_Judge_b']], category_data['cluster'])
    silhouette_scores[category] = score


silhouette_df = pd.DataFrame(list(silhouette_scores.items()), columns=['category', 'Silhouette Score'])
silhouette_df

