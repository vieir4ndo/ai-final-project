"""APRENDIZAGEM SUPERVISIONADA:
CUSTOMER SHOPPING TRENDS
Componente Curricular: Inteligência Artificial
Professor(a): Felipe Grando
Acadêmico(a): Matheus Vieira Santos"""

"""# Imports"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')

"""# Importing dataset"""

colnames = [ 'Customer ID',	'Age',	'Gender',	'Item Purchased',	'Category',	'Purchase Amount (USD)',	'Location',	'Size',	'Color',	'Season',	'Review Rating',
            'Subscription Status',	'Shipping Type',	'Discount Applied',	'Promo Code Used',	'Previous Purchases',	'Payment Method',	'Frequency of Purchases' ];
df = pd.read_csv('shopping_trends.csv', names=colnames, header=None)

"""# Showing dataset"""

df.head(48842)

"""# Data cleaning

## Dropping useless feature
"""

del df['Customer ID'];
del df['Item Purchased'];
del df['Subscription Status'];
del df['Promo Code Used'];

"""## Checking if data needs to be cleaned


"""

print('How many NaN values by attribute:')
print(df.isnull().sum())
print('How many 0 values by attribute:')
print((df==0).sum())

"""# Data transformation

## Reducing data information
"""

df = df[df['Category'] != 'Accessories']


def categorize_color(color):
    light_colors = {'Beige', 'Cyan', 'Gold', 'Lavender', 'Peach', 'Pink', 'Silver', 'White', 'Yellow'}
    dark_colors = {'Black', 'Brown', 'Charcoal', 'Gray', 'Maroon', 'Olive', 'Purple'}
    colorful_colors = {'Blue', 'Green', 'Indigo', 'Magenta', 'Orange', 'Red', 'Teal', 'Turquoise', 'Violet'}

    if color in light_colors:
        return 'Light'
    elif color in dark_colors:
        return 'Dark'
    elif color in colorful_colors:
        return 'Colorful'

df['Color'] = df['Color'].apply(categorize_color)

"""## Encoding categorical attributes as enum"""

laben = pp.LabelEncoder()

laben.fit(df['Age'])
df['Age'] = laben.transform(df['Age'])
print('Age\n', laben.classes_)

laben.fit(df['Gender'])
df['Gender'] = laben.transform(df['Gender'])
print('\nGender\n', laben.classes_)

laben.fit(df['Category'])
df['Category'] = laben.transform(df['Category'])
print('\nCategory\n', laben.classes_)

laben.fit(df['Location'])
df['Location'] = laben.transform(df['Location'])
print('\nLocation\n', laben.classes_)

laben.fit(df['Color'])
df['Color'] = laben.transform(df['Color'])
print('\nColor\n', laben.classes_)

laben.fit(df['Season'])
df['Season'] = laben.transform(df['Season'])
print('\nSeason\n', laben.classes_)

laben.fit(df['Size'])
df['Size'] = laben.transform(df['Size'])
print('\nSize\n', laben.classes_)

laben.fit(df['Frequency of Purchases'])
df['Frequency of Purchases'] = laben.transform(df['Frequency of Purchases'])
print('\nFrequency of Purchases\n', laben.classes_)

laben.fit(df['Purchase Amount (USD)'])
df['Purchase Amount (USD)'] = laben.transform(df['Purchase Amount (USD)'])
print('\nPurchase Amount (USD)\n', laben.classes_)

laben.fit(df['Shipping Type'])
df['Shipping Type'] = laben.transform(df['Shipping Type'])
print('\nShipping Type\n', laben.classes_)

laben.fit(df['Review Rating'])
df['Review Rating'] = laben.transform(df['Review Rating'])
print('\nReview Rating\n', laben.classes_)

laben.fit(df['Discount Applied'])
df['Discount Applied'] = laben.transform(df['Discount Applied'])
print('\nDiscount Applied\n', laben.classes_)

laben.fit(df['Previous Purchases'])
df['Discount Applied'] = laben.transform(df['Previous Purchases'])
print('\nPrevious Purchases\n', laben.classes_)

laben.fit(df['Payment Method'])
df['Payment Method'] = laben.transform(df['Payment Method'])
print('\nPayment Method\n', laben.classes_)

"""## Showing dataset transformed"""

df.head(20)

"""# Understanding my data"""

color = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']

plt.figure(figsize = (15,7))
value = randint(0,len(color)-4)

df.corr()
sns.heatmap(df.corr(),annot=True,cmap=color[value])
plt.title("Attribute Correlation Heat Map",fontsize = 16)
plt.tight_layout(pad=0.5)
plt.show()

"""# Data normalization"""

X = df.drop(columns=['Category'])
y = df['Category']

X.head(200)

colnames = ['Age',
            'Gender',
            'Purchase Amount (USD)',
            'Location',
            'Size',
            'Color',
            'Season',
            'Frequency of Purchases',
            'Review Rating',
            'Payment Method'];
scaler = pp.MinMaxScaler()
X[colnames] = scaler.fit_transform(X[colnames])

"""# Spliting dataset into train and test"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=145)

print("0: ",(y_train==0).sum())
print("1: ",(y_train==1).sum())
over_sampler = RandomOverSampler(random_state=100)
X_train, y_train = over_sampler.fit_resample(X_train, y_train)

print("After:")
print("0: ",(y_train==0).sum())
print("1: ",(y_train==1).sum())

"""# Using random forest"""

clf = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=250)
clf.fit(X_train, y_train)

"""# Metrics"""

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

f1_score = metrics.f1_score(y_test, clf.predict(X_test), average='weighted')
accuracy = metrics.accuracy_score(y_test, clf.predict(X_test))
precision = metrics.precision_score(y_test, clf.predict(X_test), average='weighted')
recall = metrics.recall_score(y_test, clf.predict(X_test), average='weighted')

print("F1 score: ", f1_score)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)