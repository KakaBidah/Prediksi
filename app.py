import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
from io import BytesIO
import base64
import plotly.express as px
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
        else:
            st.write("Data tidak mengandung nilai NaN.")

        # Pastikan kolom sesuai dengan format Prophet
        data['ds'] = pd.to_datetime(data['ds'])

        # Cek dan tangani outlier
        def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
            quartile1 = dataframe[col_name].quantile(q1)
            quartile2 = dataframe[col_name].quantile(q3)
            iqr = quartile2 - quartile1
            up_limit = quartile2 + 1.5 * iqr
            low_limit = quartile1 - 1.5 * iqr
            return low_limit, up_limit

        def check_outlier(dataframe, col_name):
            low_limit, up_limit = outlier_thresholds(dataframe, col_name)
            return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

        def winsorize_outliers(dataframe, col_name):
            low_limit, up_limit = outlier_thresholds(dataframe, col_name)
            dataframe[col_name] = np.where(dataframe[col_name] < low_limit, low_limit, dataframe[col_name])
            dataframe[col_name] = np.where(dataframe[col_name] > up_limit, up_limit, dataframe[col_name])

        for c in data.select_dtypes(exclude="float64").columns:
            st.write("Outlier check for {}: {}".format(c, check_outlier(data, c)))

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

        # Inisialisasi dan melatih model Prophet
        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale
        )

        with st.spinner('Model sedang melakukan prediksi...'):
            model.fit(data)
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)
        st.success('Prediksi selesai!')

        # Menampilkan hasil prediksi
        st.write('Hasil Prediksi:')
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Visualisasi prediksi dengan grafik
        st.write('Plot Prediksi:')
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.write('Komponen Tren dan Musiman:')
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        st.write('Plot Interaktif:')
        plotly_fig = plot_plotly(model, forecast)
        st.plotly_chart(plotly_fig)

        # Input tanggal awal dan akhir dari pengguna
        start_date = st.date_input("Masukkan tanggal awal untuk prediksi:")
        end_date = st.date_input("Masukkan tanggal akhir untuk prediksi:")

        # Konversi tanggal input menjadi format datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if end_date < start_date:
            st.error("Tanggal akhir tidak boleh lebih awal dari tanggal awal.")
        else:
            # Filter hasil prediksi berdasarkan rentang tanggal yang dipilih
            filtered_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
            
            # Tampilkan hasil prediksi dalam bentuk tabel
            st.write(f"Prediksi dari tanggal {start_date.date()} hingga {end_date.date()}:")
            st.table(filtered_forecast[['ds', 'yhat']])

        # Visualisasi distribusi data
        st.write('Distribusi Data:')
        st.write(data['y'].describe())
        fig3, ax = plt.subplots()
        ax.hist(data['y'], bins=20, color='skyblue', edgecolor='black')
        st.pyplot(fig3)

        # Download Hasil Prediksi
        st.write('Unduh hasil prediksi:')
        csv = forecast.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Unduh sebagai CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Download Data Asli
        st.write('Unduh data asli:')
        csv_original = data.to_csv(index=False)
        b64_original = base64.b64encode(csv_original.encode()).decode()
        href_original = f'<a href="data:file/csv;base64,{b64_original}" download="data_asli.csv">Unduh sebagai CSV</a>'
        st.markdown(href_original, unsafe_allow_html=True)

        # Menyimpan dan mengunduh grafik sebagai PNG
        st.write('Unduh grafik prediksi:')
        buf = BytesIO()
        fig1.savefig(buf, format="png")
        buf.seek(0)  # Mengatur ulang posisi buffer ke awal
        btn = st.download_button(
            label="Unduh grafik prediksi",
            data=buf,
            file_name="prediksi.png",
            mime="image/png"
        )

if st.session_state['logged_in']:
    main_app()
else:
    login()
