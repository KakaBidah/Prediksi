import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
from io import BytesIO
import base64
from datetime import timedelta

# Cek status login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    # Dummy user data
    USER_CREDENTIALS = {
        'user1': 'password1',
        'user2': 'password2'
    }
    
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful")
            st.experimental_rerun()  # Reload to show the main app
        else:
            st.error("Invalid username or password")

def main_app():
    # Tema Kustom
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5dc;
        }
        .sidebar .sidebar-content {
            background-color: #B0E57C;
        }
        .title {
            color: #FF6347;
            font-size: 2em;
        }
        .css-18e3th9 {
            color: #20B2AA;
        }
        .stImage {
            display: flex;
            justify-content: flex-end;
        }
        .stFileUploader {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    # Menambahkan logo
    logo_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Lambang_Badan_Pusat_Statistik_%28BPS%29_Indonesia.svg/1280px-Lambang_Badan_Pusat_Statistik_%28BPS%29_Indonesia.svg.png'
    st.image(logo_url, width=200)

    # Judul Aplikasi
    st.title('Prediksi Harga Barang')

    # Tempat untuk mengunggah file
    uploaded_file = st.file_uploader("Unggah file Excel Anda", type=["xlsx"])

    if uploaded_file is not None:
        # Membaca file Excel
        data = pd.read_excel(uploaded_file)
        
        # Menampilkan beberapa baris dari data
        st.write("Data yang diunggah:")
        st.write(data.head(10))

        # Validasi kolom data
        if 'ds' not in data.columns or 'y' not in data.columns:
            st.error("File Excel tidak memiliki kolom yang benar. Pastikan ada kolom 'ds' untuk tanggal dan 'y' untuk nilai.")
            st.stop()

        # Menangani nilai NaN
        if data.isnull().values.any():
            st.write("Menangani nilai NaN dalam data:")
            method = st.selectbox(
                "Pilih metode penanganan NaN:",
                ["Hapus Baris", "Isi dengan Nilai Sebelumnya", "Isi dengan Nilai Berikutnya", "Isi dengan Rata-rata", "Isi dengan Median", "Isi dengan Modus", "Isi dengan Nilai Tetap"]
            )
            if method == "Hapus Baris":
                data.dropna(inplace=True)
            elif method == "Isi dengan Nilai Sebelumnya":
                data.fillna(method='ffill', inplace=True)
            elif method == "Isi dengan Nilai Berikutnya":
                data.fillna(method='bfill', inplace=True)
            elif method == "Isi dengan Rata-rata":
                data.fillna(data.mean(), inplace=True)
            elif method == "Isi dengan Median":
                data.fillna(data.median(), inplace=True)
            elif method == "Isi dengan Modus":
                data.fillna(data.mode().iloc[0], inplace=True)
            elif method == "Isi dengan Nilai Tetap":
                fill_value = st.number_input('Masukkan nilai tetap untuk mengganti NaN:', value=0)
                data.fillna(value=fill_value, inplace=True)
            
            st.write("Data setelah penanganan NaN:")
            st.write(data.head())

        # Pastikan kolom sesuai dengan format Prophet
        data['ds'] = pd.to_datetime(data['ds'])

        # Cek dan tangani outlier
        def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
            quartile1 = dataframe[col_name].quantile(q1)
            quartile3 = dataframe[col_name].quantile(q3)
            iqr = quartile3 - quartile1
            up_limit = quartile3 + 1.5 * iqr
            low_limit = quartile1 - 1.5 * iqr
            return low_limit, up_limit

        def check_outlier(dataframe, col_name):
            low_limit, up_limit = outlier_thresholds(dataframe, col_name)
            return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

        def winsorize_outliers(dataframe, col_name):
            low_limit, up_limit = outlier_thresholds(dataframe, col_name)
            dataframe[col_name] = np.where(dataframe[col_name] < low_limit, low_limit, dataframe[col_name])
            dataframe[col_name] = np.where(dataframe[col_name] > up_limit, up_limit, dataframe[col_name])

        for c in data.select_dtypes(include="float64").columns:
            if check_outlier(data, c):
                st.write(f"Outlier ditemukan pada kolom {c}.")
                winsorize_outliers(data, c)
                st.write(f"Outlier pada kolom {c} telah ditangani.")

        # Pengaturan Model Prophet di Sidebar
        st.sidebar.header('Pengaturan Model')
        prediction_days = st.sidebar.number_input('Masukkan jumlah hari untuk prediksi:', min_value=1, max_value=365, value=90)
        seasonality_mode = st.sidebar.selectbox('Seasonality Mode', ['additive', 'multiplicative'])
        changepoint_prior_scale = st.sidebar.slider('Changepoint Prior Scale', 0.01, 0.5, 0.05)
        weekly_seasonality = st.sidebar.checkbox('Tambahkan Musiman Mingguan')
        yearly_seasonality = st.sidebar.checkbox('Tambahkan Musiman Tahunan')
        
        # Fitur tambahan
        smoothing_option = st.sidebar.selectbox('Pilih Model Penghalusan', ['None', 'Moving Average', 'Exponential Smoothing'])
        data_frequency = st.sidebar.selectbox('Frekuensi Data', ['Harian', 'Mingguan', 'Bulanan'])
        holiday_effect = st.sidebar.checkbox('Tambahkan Efek Hari Libur')
        
        # Tambahkan parameter yang lebih lanjut
        seasonality_prior_scale = st.sidebar.slider('Seasonality Prior Scale', 0.01, 10.0, 1.0)
        holidays_prior_scale = st.sidebar.slider('Holidays Prior Scale', 0.01, 10.0, 1.0)

        # Tambahkan tombol reset di bawah pengaturan model di sidebar
        if st.sidebar.button("Reset"):
            st.experimental_rerun()

        # Atur frekuensi data
        if data_frequency == 'Harian':
            freq = 'D'
        elif data_frequency == 'Mingguan':
            freq = 'W'
        else:
            freq = 'M'

        # Cek apakah frekuensi data sudah sesuai
        original_freq = pd.infer_freq(data['ds'])
        if original_freq != freq:
            data = data.set_index('ds').resample(freq).mean().reset_index()
        
        # Cek ukuran dataframe setelah perubahan frekuensi
        if data.shape[0] < 2:
            st.error("Data tidak cukup untuk membuat prediksi setelah resampling frekuensi. Harap unggah data yang lebih lengkap.")
            st.stop()

        # Terapkan smoothing jika dipilih
        if smoothing_option == 'Moving Average':
            window = st.sidebar.slider('Pilih jendela untuk Moving Average:', min_value=2, max_value=30, value=5)
            if data.shape[0] > window:
                data['y'] = data['y'].rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            else:
                st.error("Data tidak cukup untuk Moving Average dengan jendela yang dipilih.")
                st.stop()
        elif smoothing_option == 'Exponential Smoothing':
            alpha = st.sidebar.slider('Pilih alpha untuk Exponential Smoothing:', min_value=0.01, max_value=1.0, value=0.3)
            data['y'] = data['y'].ewm(alpha=alpha).mean()

        # Cek ukuran dataframe setelah smoothing
        if data.shape[0] < 2:
            st.error("Data tidak cukup untuk membuat prediksi setelah penanganan NaN dan smoothing. Harap unggah data yang lebih lengkap.")
            st.stop()

        # Inisialisasi dan melatih model Prophet
        try:
            model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale
            )

            if holiday_effect:
                # Tambahkan efek hari libur (contoh: hari libur Indonesia)
                model.add_country_holidays(country_name='ID')

            model.fit(data)

            # Membuat prediksi
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)

            # Menampilkan grafik hasil prediksi
            st.subheader('Hasil Prediksi')
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig)

            # Menyediakan opsi untuk mengunduh hasil prediksi
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(forecast)
            st.download_button(
                label="Unduh Hasil Prediksi",
                data=csv,
                file_name='forecast.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melatih model: {e}")

if st.session_state['logged_in']:
    main_app()
else:
    login()
