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
    
    st.write("##### Nama         : Muhammad Habibur Rohman")
    st.write("##### NIM          : 220411100079")
    st.write("##### Mata Kuliah  : Proyek Sain Data")
    st.write("##### Kelas        : IF5B")
    st.write("""
             Air minum merupakan kebutuhan dasar manusia yang sangat penting untuk mendukung kehidupan sehari-hari. 
             Kualitas air minum yang baik menjadi salah satu syarat utama untuk menjaga kesehatan masyarakat. Namun, dengan 
             meningkatnya aktivitas manusia dan perubahan lingkungan, kualitas air minum sering kali mengalami penurunan, 
             baik dari segi kimia, fisik, maupun biologis. Kehidupan manusia bergantung pada air, tetapi berbagai masalah 
             lingkungan dan kesehatan terjadi akibat kualitas air yang buruk. Penyebab aksesibilitas air masih terbatas 
             karena pembangunan yang tidak memperhatikan keseimbangan wilayah dan mengurangi daerah resapan terutama di daerah 
             perkotaan. Akibatnya, sedikit sumber air bersih yang tersedia. Kontaminasi air merupakan masalah yang signifikan di Indonesia. 
             """)
    
    
    # Tampilan 5 Data Awal
    st.write("### Tampilan 5 Data Awal")
    st.dataframe(df.head())

    # # Informasi Atribut Dataset
    # st.write("### Informasi Atribut Dataset")
    # buffer = df.info()
    # st.text(buffer)
    
    # Pengertian Atribut
    st.write("### Pengertian Atribut")
    st.write("""
    Penjelasan atribut yang digunakan dalam klasifikasi data kualitas air : 
    1.	pH (Keasaman): Mengukur tingkat keasaman atau kebasaan air. Air dengan pH terlalu rendah (asam) atau terlalu tinggi (basa) dapat berbahaya bagi kesehatan. Standar pH untuk air minum biasanya berada di kisaran 6,5–8,5.
    2.	Hardness (Kekerasan Air): Kekerasan air diukur berdasarkan kandungan mineral, terutama kalsium dan magnesium. Air yang terlalu keras dapat menyebabkan penumpukan kerak pada pipa, sementara air yang terlalu lunak dapat menimbulkan rasa yang tidak enak.
    3.	Total Dissolved Solids (TDS): Mengukur jumlah total zat padatan terlarut dalam air, seperti garam dan mineral. Standar TDS untuk air minum biasanya tidak melebihi 500 mg/L.
    4.	Chloramines: Senyawa yang terbentuk dari reaksi klorin dengan amonia digunakan sebagai desinfektan dalam pengolahan air. Kandungan chloramines yang tinggi dapat menimbulkan bau dan rasa tidak sedap pada air.
    5.	Sulfate: Kandungan sulfat yang berlebihan dapat menyebabkan diare atau gangguan pencernaan. Standar sulfat dalam air minum umumnya tidak melebihi 250 mg/L.
    6.	Conductivity (Konduktivitas): Mengukur kemampuan air menghantarkan listrik, yang menunjukkan tingkat ion terlarut dalam air.
    7.	Organic Carbon: Menunjukkan keberadaan karbon organik dalam air, yang berasal dari bahan organik terlarut. Kadar yang tinggi dapat menunjukkan kontaminasi biologis atau bahan organik lainnya.
    8.	Trihalomethanes (THMs): Senyawa yang terbentuk selama proses desinfeksi air. Paparan jangka panjang terhadap THMs dapat meningkatkan risiko gangguan kesehatan.
    9.	Turbidity (Kekeruhan): Menunjukkan jumlah partikel tersuspensi dalam air. Kekeruhan yang tinggi dapat menjadi indikator adanya kontaminasi biologis atau partikel lain yang berbahaya.
    """)

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
    st.write("##### Nama         : Muhammad Habibur Rohman")
    st.write("##### NIM          : 220411100079")
    st.write("##### Mata Kuliah  : Proyek Sain Data")
    st.write("##### Kelas        : IF5B")

    # Menampilkan informasi missing values
    st.write("### Missing Values Data Kualitas Air")
    st.write("""Pengertian Missing value adalah masalah yang muncul dalam kualitas data.
             Missing value disebabkan oleh beberapa faktor seperti data yang membutuhkan suatu penyimpanan besar atau data yang 
             memiliki ukuran yang sangat besar. Masalah utama missing value yaitu susah dalam pencarian. Untuk menanggani masalah 
             tersebut digunakan metode imputasi.""")
    
    missing_values = df.isnull().sum()
    st.write("Jumlah missing values per kolom:")
    st.write(missing_values[missing_values > 0])

    # Tampilkan 5 data dengan missing values
    st.write("##### Data dengan Missing Values")
    missing_rows = df[df.isnull().any(axis=1)]
    st.write(missing_rows.head(5))

    # Mengisi missing values dengan rata-rata
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    st.write("##### Data Setelah Diisi Missing Values")
    st.write(df.head(5))

    # Mengecek dan menampilkan informasi tentang outlier
    st.write("### Outlier Detection")
    numeric_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    # Menghitung Z-score untuk setiap fitur numerik
    z_scores = np.abs(df[numeric_features].apply(zscore))

    # Menentukan batas Z-score untuk mendeteksi outlier
    outliers_mask = z_scores > 3

    # Visualisasi Outlier Sebelum dan Sesudah Penanganan untuk Setiap Atribut
    st.write("### Visualisasi Outlier Sebelum dan Sesudah Penanganan untuk Setiap Atribut")
    plt.figure(figsize=(15, 20))

    for i, feature in enumerate(numeric_features):
        # Sebelum Penanganan
        plt.subplot(len(numeric_features), 2, i * 2 + 1)
        x = df.index
        y = df[feature]
        outlier_indices = outliers_mask[feature]
        plt.scatter(x, y, alpha=0.5, c='blue', label='Data')
        plt.scatter(
            x[outlier_indices], 
            y[outlier_indices], 
            alpha=0.7, c='red', label='Outliers'
        )
        plt.title(f"{feature} - Sebelum Penanganan")
        plt.xlabel("Index")
        plt.ylabel(feature)
        plt.legend()

        # Setelah Penanganan
        plt.subplot(len(numeric_features), 2, i * 2 + 2)
        df_no_outliers = df[(z_scores < 3).all(axis=1)]  # Menghapus outlier
        x_cleaned = df_no_outliers.index
        y_cleaned = df_no_outliers[feature]
        plt.scatter(x_cleaned, y_cleaned, alpha=0.5, c='green', label='Data Cleaned')
        plt.title(f"{feature} - Setelah Penanganan")
        plt.xlabel("Index")
        plt.ylabel(feature)
        plt.legend()

    plt.tight_layout()
    st.pyplot(plt)

    # Data setelah penanganan outlier
    df_no_outliers = df[(z_scores < 3).all(axis=1)]

    # Menampilkan data setelah penghapusan outlier
    st.write("##### Data Setelah Menghapus Outlier")
    st.write(df_no_outliers.head(5))

    # Proses balancing data menggunakan SMOTE
    st.write("### Pre-processing (Penyeimbangan Data)")
    st.write("### Pengertian SMOTE:")
    st.write("""Imbalanced data merupakan data set dengan kelas yang tidak seimbang. 
             Biasanya untuk data seperti peneliti diperlukan penangganan. Salah satu 
             penangganan adalah dengan melakukan SMOTE yaitu melakukan perhitungan untuk menambahkan 
             dataset dengan jumlah yang seimbang.""")

    # Asumsikan 'Potability' adalah kolom target
    X = df_no_outliers.drop(columns=['Potability'])
    y = df_no_outliers['Potability']

    # Sebelum balancing
    st.write("##### Grafik Sebelum Penyeimbangan Data")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Distribusi Data Sebelum Penyeimbangan")
    st.pyplot(plt)

    # Penyeimbangan data dengan SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Setelah balancing
    st.write("##### Grafik Setelah Penyeimbangan Data dengan SMOTE")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_resampled)
    plt.title("Distribusi Data Setelah Penyeimbangan")
    st.pyplot(plt)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Modelling
elif menu == "Modelling":
    st.title("Modelling - Klasifikasi Kualitas Air Menggunakan Metode Decision Tree C.45")
    st.write("##### Nama         : Muhammad Habibur Rohman")
    st.write("##### NIM          : 220411100079")
    st.write("##### Mata Kuliah  : Proyek Sain Data")
    st.write("##### Kelas        : IF5B")
    
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

    # Visualisasi Distribusi Target pada Data Training dan Testing
    st.write("### Visualisasi Pembagian Data Training dan Testing")

    # Grafik untuk Data Training
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Training data distribution
    sns.countplot(x=y_train, ax=ax[0])
    ax[0].set_title('Distribusi Target pada Data Training')
    ax[0].set_xlabel('Potability')
    ax[0].set_ylabel('Count')

    # Testing data distribution
    sns.countplot(x=y_test, ax=ax[1])
    ax[1].set_title('Distribusi Target pada Data Testing')
    ax[1].set_xlabel('Potability')
    ax[1].set_ylabel('Count')

    st.pyplot(fig)

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
    
    # Visualisasi pohon keputusan
    st.write("### Visualisasi Pohon Keputusan")
    fig = plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        filled=True,
        feature_names=X.columns,
        class_names=['Not Potable', 'Potable'],
        rounded=True,
        proportion=True,
        fontsize=10,
    ) 
    st.pyplot(plt)
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Klasifikasi
elif menu == "Klasifikasi":
    st.title("Klasifikasi - Input")
    st.write("##### Nama         : Muhammad Habibur Rohman")
    st.write("##### NIM          : 220411100079")
    st.write("##### Mata Kuliah  : Proyek Sain Data")
    st.write("##### Kelas        : IF5B")
    
    # Muat model yang sudah dilatih
    model = joblib.load('kualitas_air_model.pkl')  # Memuat model yang disimpan sebelumnya
    
    # Masukkan data untuk prediksi
    st.write("Masukkan data untuk prediksi:")
    input_data = {}
    numeric_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                       'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    # Input data
    for col in df.drop('Potability', axis=1).columns:
        if col in numeric_columns:
            # Numerical input with number_input, default None (placeholder kosong)
            input_data[col] = st.number_input(f"{col}:", value=None, step=0.01)
        else:
            # Text input for other types
            input_data[col] = st.text_input(f"{col}:", value="", placeholder=f"Masukkan {col}")

    if st.button("Prediksi"):
        # Validasi jika ada input kosong
        if None in input_data.values() or "" in input_data.values():
            st.error("Harap isi semua nilai untuk melakukan prediksi.")
        else:
            # Membuat DataFrame dari input pengguna
            input_df = pd.DataFrame([input_data])
            
            # Prediksi menggunakan model yang dimuat
            prediction = model.predict(input_df)[0]
            
            # Menampilkan hasil prediksi
            result = "Air bisa diminum" if prediction == 1 else "Tidak bisa diminum"
            st.write(f"Hasil Prediksi: {result}")
