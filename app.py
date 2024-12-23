import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import joblib

# Set halaman Streamlit
st.set_page_config(page_title="Klasifikasi Air Minum Menggunakan Metode Decision Tree C.45", layout="wide")

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    return pd.read_csv('water_potability.csv')

# Memuat dataset
df = load_data()

# Sidebar navigasi
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Analisis Data", "Pre-Processing", "Modelling", "Klasifikasi"])

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Analisis Data
if menu == "Analisis Data":
    st.title("Analisis Data - Klasifikasi Air Minum Menggunakan Metode Decision Tree C.45")
    
    # Tampilan 5 Data Awal
    st.write("### Tampilan 5 Data Awal")
    st.dataframe(df.head())

    # Informasi Atribut Dataset
    st.write("### Informasi Atribut Dataset")
    buffer = df.info()
    st.text(buffer)

    # Mengecek Missing Value
    st.write("### Mengecek Missing Values")
    missing_values = df.isnull().sum()
    st.table(missing_values[missing_values > 0])

    # Analisis Statistik
    st.write("### Analisis Statistik")
    st.dataframe(df.describe())

    # Distribusi Target
    st.write("### Distribusi Target")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Potability', data=df)
    plt.title("Distribusi Air Minum Outcome")
    plt.xlabel("Potability Air Minum")
    plt.ylabel("Count")
    st.pyplot(plt)

    stroke_counts = df['Potability'].value_counts()
    stroke_distribution = df['Potability'].value_counts(normalize=True) * 100
    stroke_summary = pd.DataFrame({
        'Count': stroke_counts,
        'Percentage (%)': stroke_distribution
    })
    st.write("Distribusi Target:")
    st.table(stroke_summary)

    # Distribusi Fitur Numerik
    st.write("### Distribusi Fitur Numerik")
    first_row_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate']
    second_row_features = ['Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    # Ukuran figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))

    # Plot untuk baris pertama
    for i, feature in enumerate(first_row_features):
        df[feature].hist(bins=20, ax=axes[0, i])
        axes[0, i].set_title(feature)

    # Plot untuk baris kedua
    for i, feature in enumerate(second_row_features):
        df[feature].hist(bins=20, ax=axes[1, i])
        axes[1, i].set_title(feature)
    for j in range(len(second_row_features), 5):
        fig.delaxes(axes[1, j])
    
    plt.suptitle("Distribution of Numerical Features")
    st.pyplot(plt)

    # Mengecek Outlier menggunakan Z-Score
    st.write("### Mengecek Outlier Menggunakan Z-Score")
    numeric_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    # Menghitung Z-score untuk setiap fitur numerik
    z_scores = np.abs(df[numeric_features].apply(zscore))

    # Menentukan batas Z-score untuk mendeteksi outlier (misalnya Z-score > 3)
    outliers = (z_scores > 3).sum(axis=0)

    # Menampilkan fitur yang memiliki outlier
    outliers_summary = pd.DataFrame({
        'Feature': numeric_features,
        'Outliers': outliers
    })
    # Menampilkan tabel jumlah outlier per fitur
    st.table(outliers_summary)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Pre-Processing
elif menu == "Pre-Processing":
    st.title("Pre-Processing - Klasifikasi Air Minum Menggunakan Metode Decision Tree C.45")

    # Menampilkan informasi missing values
    st.write("### Missing Values Data Kualitas Air")
    missing_values = df.isnull().sum()
    st.write("Jumlah missing values per kolom:")
    st.write(missing_values[missing_values > 0])

    # Tampilkan 5 data dengan missing values
    st.write("##### Data dengan Missing Values")
    missing_rows = df[df.isnull().any(axis=1)]  # Mendapatkan baris yang memiliki missing values

    st.write("Menampilkan 5 baris data :")
    st.write(missing_rows.head(5))

    # Mengisi missing values dengan rata-rata
    for col in missing_rows.columns:
        if missing_rows[col].isnull().sum() > 0:  # Pastikan kolom ini memiliki missing values
            df[col].fillna(df[col].mean(), inplace=True)

    # Tampilkan 5 data setelah diisi missing values
    st.write("##### Data Setelah Diisi Missing Values (Menggunakan Nilai rata-rata)")
    st.write("Menampilkan 5 baris data :")
    st.write(df.head(5))

    # Menampilkan informasi missing values setelah pengisian
    st.write("##### Informasi Missing Values Setelah Pengisian")
    remaining_missing_values = df.isnull().sum()
    st.write("Jumlah missing values per kolom setelah pengisian:")
    st.write(remaining_missing_values)

    # Mengecek Outlier menggunakan Z-Score
    st.write("### Mengecek Outlier Menggunakan Z-Score")
    numeric_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    # Menghitung Z-score untuk setiap fitur numerik
    z_scores = np.abs(df[numeric_features].apply(zscore))

    # Menentukan batas Z-score untuk mendeteksi outlier (misalnya Z-score > 3)
    outliers = (z_scores > 3).sum(axis=0)

    # Menampilkan fitur yang memiliki outlier
    outliers_summary = pd.DataFrame({
        'Feature': numeric_features,
        'Outliers': outliers
    })

    # Menampilkan tabel jumlah outlier per fitur
    st.table(outliers_summary)

    # Menghapus outlier (menggunakan Z-score > 3 sebagai batas)
    df_no_outliers = df[(z_scores < 3).all(axis=1)]

    # Tampilkan 5 data setelah penghapusan outlier
    st.write("##### Data Setelah Menghapus Outlier")
    st.write("Menampilkan 5 baris data :")
    st.write(df_no_outliers.head(5))

    # Asumsikan 'Potability' adalah kolom target (binary: 0 atau 1)
    # Pisahkan fitur dan target
    X = df_no_outliers.drop(columns=['Potability'])
    y = df_no_outliers['Potability']

    # Tampilkan grafik distribusi target sebelum balancing
    st.write("### Pre-processing ( Penyeimbangan Data )")
    st.write("##### Grafik Sebelum data seimbang")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Penyeimbangan Data")
    plt.ylim(0, 700)  # Mengatur rentang sumbu y dari 0 hingga 700
    st.pyplot(plt)

    # Tampilkan tabel jumlah data sebelum seimbang
    st.write("##### Jumlah Data Sebelum Penyeimbangan")
    before_balance = pd.DataFrame(y.value_counts()).reset_index()
    before_balance.columns = ['Potability', 'Count']
    st.write(before_balance)

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menangani imbalance data dengan SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Tampilkan grafik distribusi target setelah balancing menggunakan SMOTE
    st.write("##### Grafik penyeimbangan data dengan SMOTE")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_resampled)
    plt.title("Penyeimbangan Data")
    plt.ylim(0, 700)  # Mengatur rentang sumbu y dari 0 hingga 700
    st.pyplot(plt)

    # Tampilkan tabel jumlah data setelah seimbang
    st.write("##### Jumlah Data Setelah Penyeimbangan dengan SMOTE")
    after_balance = pd.DataFrame(y_resampled.value_counts()).reset_index()
    after_balance.columns = ['Potability', 'Count']
    st.write(after_balance)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Modelling
elif menu == "Modelling":
    st.title("Modelling - Klasifikasi Kualitas Air Menggunakan Metode Decision Tree C.45")
    
    # Menampilkan informasi missing values
    missing_values = df.isnull().sum()

    # Tampilkan 5 data dengan missing values
    missing_rows = df[df.isnull().any(axis=1)]  # Mendapatkan baris yang memiliki missing values

    # Mengisi missing values dengan rata-rata
    for col in missing_rows.columns:
        if missing_rows[col].isnull().sum() > 0:  # Pastikan kolom ini memiliki missing values
            df[col].fillna(df[col].mean(), inplace=True)

    # Mengecek Outlier menggunakan Z-Score
    numeric_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    # Menghitung Z-score untuk setiap fitur numerik
    z_scores = np.abs(df[numeric_features].apply(zscore))

    # Menentukan batas Z-score untuk mendeteksi outlier (misalnya Z-score > 3)
    outliers = (z_scores > 3).sum(axis=0)

    # Menampilkan fitur yang memiliki outlier
    outliers_summary = pd.DataFrame({
        'Feature': numeric_features,
        'Outliers': outliers
    })

    # Menghapus outlier (menggunakan Z-score > 3 sebagai batas)
    df_no_outliers = df[(z_scores < 3).all(axis=1)]

    # Asumsikan 'Potability' adalah kolom target (binary: 0 atau 1)
    # Pisahkan fitur dan target
    X = df_no_outliers.drop(columns=['Potability'])
    y = df_no_outliers['Potability']

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tampilkan informasi tentang pembagian data training dan testing
    st.write("### Pembagian Data Training dan Testing")
    
    # Menampilkan informasi jumlah data
    st.write(f"Jumlah Data Training: {X_train.shape[0]} sampel, {X_train.shape[1]} fitur")
    st.write(f"Jumlah Data Testing: {X_test.shape[0]} sampel, {X_test.shape[1]} fitur")

    # Menampilkan 5 data pertama dari Training set
    st.write("#### 5 Data Pertama Training Set")
    st.write(X_train.head())  # Menampilkan 5 data pertama dari fitur training

    # Menampilkan 5 data pertama dari target training set
    st.write("#### 5 Target Pertama Training Set")
    st.write(y_train.head())  # Menampilkan 5 data target training

    # Menampilkan 5 data pertama dari Testing set
    st.write("#### 5 Data Pertama Testing Set")
    st.write(X_test.head())  # Menampilkan 5 data pertama dari fitur testing

    # Menampilkan 5 data pertama dari target testing set
    st.write("#### 5 Target Pertama Testing Set")
    st.write(y_test.head())  # Menampilkan 5 data target testing
    
    # Menangani imbalance data dengan SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Tampilkan jumlah data setelah penyeimbangan menggunakan SMOTE
    st.write(f"Jumlah Data Setelah SMOTE: {X_resampled.shape[0]} sampel")

    # Menampilkan jumlah data target setelah SMOTE dalam format tabel
    smote_count = pd.DataFrame(y_resampled.value_counts()).reset_index()
    smote_count.columns = ['Potability', 'Count']
    st.write("##### Jumlah Data Target Setelah SMOTE:")
    st.write(smote_count)
    
    # MASUK KEDALAM METODE C45
    model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=2)  # C4.5 menggunakan 'entropy' sebagai kriteria
    model.fit(X_resampled, y_resampled)

    # Memprediksi target untuk data testing
    y_pred = model.predict(X_test)
    # Menyimpan model setelah pelatihan
    joblib.dump(model, 'kualitas_air_model.pkl')  # Simpan model

    # Menghitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Akurasi Model: {accuracy * 100:.2f}%")

    # Menampilkan confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # Menampilkan classification report
    st.write("### Classification Report")
    st.write(classification_report(y_test, y_pred))

    # Visualisasi pohon keputusan (optional)
    st.write("### Visualisasi Pohon Keputusan")
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Potable', 'Potable'], rounded=True)
    st.pyplot(plt)
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Klasifikasi
elif menu == "Klasifikasi":
    st.title("Klasifikasi - Input")
    
    # Muat model yang sudah dilatih
    model = joblib.load('kualitas_air_model.pkl')  # Memuat model yang disimpan sebelumnya
    
    # Masukkan data untuk prediksi
    st.write("Masukkan data untuk prediksi:")
    input_data = {}
    categorical_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']  # Adjust categorical columns based on your dataset

    # Handle categorical and numerical features separately
    for col in df.drop('Potability', axis=1).columns:
        if col in categorical_columns:
            # Categorical input with selectbox
            categories = df[col].unique().tolist()
            input_data[col] = st.selectbox(f"{col}:", categories, index=categories.index(df[col].mode()[0]))
        else:
            # Numerical input with number_input and initial value set to None (0 or empty)
            input_data[col] = st.number_input(f"{col}:", value=0.0)  # Initial value 0.0 as a placeholder

    if st.button("Prediksi"):
        # Membuat DataFrame dari input pengguna
        input_df = pd.DataFrame([input_data])
        
        # Prediksi menggunakan model yang dimuat
        prediction = model.predict(input_df)[0]
        
        # Menampilkan hasil prediksi
        result = "Air bisa diminum" if prediction == 1 else "Tidak bisa diminum"
        st.write(f"Hasil Prediksi: {result}")
