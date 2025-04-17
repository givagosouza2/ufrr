import streamlit as st
import pandas as pd
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def data_processing(t, x, y, z, freq):
    # Garantir que t seja um array numpy
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Evitar divisão por zero em novo_t
    if t[-1] <= 0:
        t[-1] = 1

    novo_t = np.linspace(0, t[-1], int(len(t) * (0.1 / (len(t) / t[-1]))))

    # Remover tendências
    x_detrended = signal.detrend(x)
    y_detrended = signal.detrend(y)
    z_detrended = signal.detrend(z)

    # Interpolação
    x_interp = np.interp(novo_t, t, x_detrended)
    y_interp = np.interp(novo_t, t, y_detrended)
    z_interp = np.interp(novo_t, t, z_detrended)

    # Filtro passa-baixa
    fs = 100  # Frequência de amostragem
    cutoff = freq  # Frequência de corte
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low')

    x_filtered = signal.filtfilt(b, a, x_interp)
    y_filtered = signal.filtfilt(b, a, y_interp)
    z_filtered = signal.filtfilt(b, a, z_interp)

    # Norma
    norma = np.sqrt(x_filtered**2 + y_filtered**2 + z_filtered**2)
    return novo_t, norma


st.set_page_config(layout="wide")

# Título do aplicativo
st.title("TUG instrumentado")

c1, c2 = st.columns(2)
c_acc = 0
c_gyro = 0

with c1:
    uploaded_file_acc = st.file_uploader(
        "Escolha o arquivo do acelerômetro", type=["txt"])

    if uploaded_file_acc is not None:
        # Leitura do arquivo
        data_acc = pd.read_csv(uploaded_file_acc, sep=";")

        # Separando as colunas em variáveis
        tempo_acc = data_acc.iloc[:, 0]  # Primeira coluna
        x_acc = data_acc.iloc[:, 1]  # Segunda coluna
        y_acc = data_acc.iloc[:, 2]  # Terceira coluna
        z_acc = data_acc.iloc[:, 3]  # Quarta coluna
        c_acc = 1


with c2:
    uploaded_file_gyro = st.file_uploader(
        "Escolha o arquivo do giroscópio", type=["txt"])

    if uploaded_file_gyro is not None:
        # Leitura do arquivo
        data_gyro = pd.read_csv(uploaded_file_gyro, sep=";")

        # Separando as colunas em variáveis
        tempo_gyro = data_gyro.iloc[:, 0]  # Primeira coluna
        x_gyro = data_gyro.iloc[:, 1]  # Segunda coluna
        y_gyro = data_gyro.iloc[:, 2]  # Terceira coluna
        z_gyro = data_gyro.iloc[:, 3]  # Quarta coluna

        c_gyro = 1

if c_acc == 1 and c_gyro == 1:
    tempo_acc_proc, norma_acc = data_processing(
        tempo_acc, x_acc, y_acc, z_acc, 6)
    tempo_gyro_proc, norma_gyro = data_processing(
        tempo_gyro, x_gyro, y_gyro, z_gyro, 1.5)
    with c1:
        onset = st.slider("Marque o início do teste no giroscópio",
                          min_value=0, max_value=len(tempo_gyro_proc), value=0)
        G1 = st.slider("Marque o componente G1 teste no giroscópio",
                       min_value=0, max_value=len(tempo_gyro_proc), value=0)
        G2 = st.slider("Marque o componente G2 teste no giroscópio",
                       min_value=0, max_value=len(tempo_gyro_proc), value=0)
    with c2:
        A1 = st.slider("Marque o componente A1 teste no acelerômetro",
                       min_value=0, max_value=len(tempo_acc_proc), value=0)
        A2 = st.slider("Marque o componente A2 teste no acelerômetro",
                       min_value=0, max_value=len(tempo_acc_proc), value=0)
        offset = st.slider("Marque o fim do teste no acelerômetro",
                           min_value=0, max_value=len(tempo_acc_proc), value=0)

    with c1:
        fig, ax = plt.subplots()
        ax.plot(tempo_acc_proc, norma_acc)
        ax.plot([tempo_gyro_proc[onset],
                tempo_gyro_proc[onset]], [0, 10], '--r')
        ax.plot([tempo_gyro_proc[G1],
                tempo_gyro_proc[G1]], [0, 10], '--g')
        ax.plot([tempo_gyro_proc[G2],
                tempo_gyro_proc[G2]], [0, 10], '--g')
        ax.plot([tempo_acc_proc[offset],
                tempo_acc_proc[offset]], [0, 10], '--r')
        ax.plot([tempo_acc_proc[A1],
                tempo_acc_proc[A1]], [0, 10], '--b')
        ax.plot([tempo_acc_proc[A2],
                tempo_acc_proc[A2]], [0, 10], '--b')
        
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Aceleração (m/s^2)')
        ax.set_title('Norma do Acelerômetro')
        st.pyplot(fig)

    with c2:
        if c_gyro == 1:
            fig, ax = plt.subplots()
            tempo_gyro_proc = tempo_gyro_proc
            ax.plot(tempo_gyro_proc, norma_gyro,'k')
            ax.plot([tempo_gyro_proc[onset],
                     tempo_gyro_proc[onset]], [0, 5], '--r')
            ax.plot([tempo_gyro_proc[G1],
                    tempo_gyro_proc[G1]], [0, 5], '--g')
            ax.plot([tempo_gyro_proc[G2],
                    tempo_gyro_proc[G2]], [0, 5], '--g')
            ax.plot([tempo_gyro_proc[offset],
                    tempo_gyro_proc[offset]], [0, 5], '--r')
            ax.plot([tempo_acc_proc[A1],tempo_acc_proc[A1]], [0, 5], '--b')
            ax.plot([tempo_acc_proc[A2],tempo_acc_proc[A2]], [0, 5], '--b')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angular velocity (rad/s)')
            ax.set_title('Norma do Giroscópio')
            st.pyplot(fig)
    st.text('Duração total (s) = ' +
            str(round((tempo_acc_proc[offset] - tempo_gyro_proc[onset])/1000,2)))
    st.text('Duração de sentar para levantar (s) = ' +
            str(round((tempo_acc_proc[A1]-tempo_gyro_proc[onset])/1000,2)))
    st.text('Duração da caminhada de ida (s) = ' +
            str(round((tempo_gyro_proc[G1] - tempo_acc_proc[A1])/1000,2)))
    st.text('Duração da caminhada de retorno (s) = ' +
            str(round((tempo_gyro_proc[G2] - tempo_gyro_proc[G1])/1000,2)))
    st.text('Duração de em pé para sentar (s) = ' +
            str(round((tempo_acc_proc[offset] - tempo_acc_proc[A2])/1000,2)))
    st.text('Pico de A1 (m/s2) = ' + str(norma_acc[A1]))
    st.text('Pico de A2 (m/s2) = ' + str(norma_acc[A2]))
    st.text('Pico de G1 (rad/s) = ' + str(norma_gyro[G1]))
    st.text('Pico de G2 (rad/s) = ' + str(norma_gyro[G2]))
