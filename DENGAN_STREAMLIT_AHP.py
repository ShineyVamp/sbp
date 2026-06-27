import streamlit as st
import pandas as pd
import numpy as np
import io
import sqlite3
import time
from scipy.optimize import linear_sum_assignment
    
#database
def init_db():
    conn = sqlite3.connect('penjurusan.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS hasil_penjurusan (
            NIS TEXT PRIMARY KEY,
            Nama TEXT,
            Kelas_Lama TEXT,
            Psikotes TEXT,
            Minat TEXT,
            Rata_Alam REAL,
            Rata_Sosial REAL,
            Rata_Matematika REAL,
            Jurusan_1 TEXT,
            Jurusan_2 TEXT,
            Rekomendasi_Sistem TEXT,
            Alasan_Rekomendasi TEXT,
            Verifikasi_Guru TEXT
        )
    ''')
    try:
        c.execute("ALTER TABLE hasil_penjurusan ADD COLUMN Ortu TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass # Kolom sudah ada
    conn.commit()
    conn.close()

init_db()

st.set_page_config(
    page_title="SPK Penjurusan Kelas",
    layout="wide",
    initial_sidebar_state="expanded"
)

#bobot kelas
PAKET_KELAS = {
    "F1":  {"Alam": 2, "Sosial": 4, "Matematika": 2, "Teknik": 2},
    "F2":  {"Alam": 2, "Sosial": 5, "Matematika": 2, "Teknik": 2},
    "F3":  {"Alam": 4, "Sosial": 2, "Matematika": 5, "Teknik": 3},
    "F4":  {"Alam": 4, "Sosial": 2, "Matematika": 5, "Teknik": 3},
    "F5":  {"Alam": 4, "Sosial": 2, "Matematika": 3, "Teknik": 4},
    "F6":  {"Alam": 4, "Sosial": 2, "Matematika": 3, "Teknik": 4},
    "F7":  {"Alam": 4, "Sosial": 3, "Matematika": 3, "Teknik": 2},
    "F8":  {"Alam": 4, "Sosial": 3, "Matematika": 3, "Teknik": 2},
    "F9":  {"Alam": 4, "Sosial": 2, "Matematika": 3, "Teknik": 2},
    "F10": {"Alam": 4, "Sosial": 2, "Matematika": 3, "Teknik": 2}
}

KUOTA_KELAS = {
    "F1": 40, "F2": 40, "F3": 40, "F4": 40, "F5": 40,
    "F6": 40, "F7": 40, "F8": 40, "F9": 40, "F10": 40
}

KELAS_MAT_LANJUT = ["F3", "F4"]
AMBANG_MATEMATIKA_MAT_LANJUT = 86

KOLOM_UPLOAD = ["NIS", "Nama", "Psikotes", "Minat", "Rata_Alam", "Rata_Sosial",
                "Rata_Matematika", "KULIAH JURUSAN 1", "KULIAH JURUSAN 2"] 

KOLOM_EDITOR = ["NIS", "Nama", "Kelas_Lama", "Psikotes", "Minat", "Rata_Alam",
                "Rata_Sosial", "Rata_Matematika", "KULIAH JURUSAN 1", "KULIAH JURUSAN 2",
                "Rekomendasi_Sistem", "Alasan_Rekomendasi", "Verifikasi_Guru"]

KOLOM_NUMERIK = ["Rata_Alam", "Rata_Sosial", "Rata_Matematika"]

# AHP
NAMA_KRITERIA_AHP = ["Rata Alam", "Rata Sosial", "Rata Matematika", "Psikotes", "Minat",
                     "Relevansi Jurusan 1", "Relevansi Jurusan 2"]
N_KRITERIA = 7
RI_TABLE = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# Bobot default (%). Inilah sumber kebenaran. Matriks Saaty diturunkan dari sini.
# Relevansi jurusan dipisah: Jurusan 1 (pilihan utama) diberi bobot lebih besar dari Jurusan 2.
BOBOT_DEFAULT_PERSEN = np.array([15.0, 15.0, 15.0, 7.0, 10.0, 25.0, 13.0])

SAATY_OPTIONS = {
    "9 — A jauh lebih penting": 9, "7 — A sangat lebih penting": 7, "5 — A cukup lebih penting": 5,
    "3 — A sedikit lebih penting": 3, "1 — Sama penting": 1,
    "1/3 — B sedikit lebih penting": 1/3, "1/5 — B cukup lebih penting": 1/5,
    "1/7 — B sangat lebih penting": 1/7, "1/9 — B jauh lebih penting": 1/9,
}

def hitung_bobot_ahp(matrix):
    n = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    norm = matrix / col_sums
    bobot = norm.mean(axis=1)
    Aw = matrix @ bobot
    lambda_max = np.mean(Aw / bobot)
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI_TABLE.get(n, 1.49)
    return bobot, CR, lambda_max

def matriks_dari_bobot(w):
    # Bangun matriks perbandingan berpasangan dari bobot: A[i][j] = w_i / w_j.
    # Matriks ini konsisten sempurna (CR = 0) karena dibangun dari rasio bobot.
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return np.outer(w, 1.0 / w)

def nilai_ke_label_saaty(v):
    candidates = {abs(v - val): lbl for lbl, val in SAATY_OPTIONS.items()}
    return candidates[min(candidates)]


#mapping jurusan dan alasan
def dapatkan_rumpun_ilmu(teks_jurusan):
    teks = str(teks_jurusan).upper().strip()
    if not teks or teks == "NAN":
        return "UMUM"
    if any(k in teks for k in ["STAN", "STIS", "IPDN", "AKPOL", "AKMIL", "STMKG", "STIN", "KEDINASAN"]):
        return "KEDINASAN"
    if any(k in teks for k in ["ISLAM", "TEOLOGI", "AGAMA"]):
        return "AGAMA"
    if any(k in teks for k in ["SASTRA", "SEJARAH", "FILSAFAT", "BUDAYA", "SENI"]):
        return "HUMANIORA"
    if any(k in teks for k in ["MATEMATIKA", "STATISTIKA", "ILMU KOMPUTER", "KOMPUTER"]):
        return "FORMAL"
    if any(k in teks for k in ["KIMIA", "FISIKA", "BIOLOGI", "GEOGRAFI", "SAINS"]) and "TEKNIK" not in teks:
        return "ALAM"
    if any(k in teks for k in ["TEKNIK", "ARSITEKTUR", "DESAIN", "SIPIL", "MESIN"]):
        return "TERAPAN_TEKNIK"
    if any(k in teks for k in ["PERTANIAN", "PETERNAKAN", "AGRIBISNIS", "AGRO"]):
        return "TERAPAN_AGRO"
    if any(k in teks for k in ["KEDOKTERAN", "MEDIS", "PSIKOLOGI", "FARMASI", "KEPERAWATAN", "KESEHATAN"]):
        return "TERAPAN_MEDIS"
    if any(k in teks for k in ["AKUNTANSI", "BISNIS", "MANAJEMEN", "HUKUM", "PENDIDIKAN"]):
        return "TERAPAN_SOSIAL"
    if any(k in teks for k in ["SOSIOLOGI", "POLITIK", "EKONOMI"]):
        return "SOSIAL"
    return "UMUM"

# Kelompok kelas berdasarkan karakter profilnya. Dipakai untuk skor relevansi.
KELAS_SOSIAL      = ["F1", "F2"]    # condong sosial
KELAS_SAINS_MAT   = ["F3", "F4"]    # sains + matematika kuat
KELAS_SAINS_TEK   = ["F5", "F6"]    # sains + terapan/teknik
KELAS_SAINS_KES   = ["F7", "F8"]    # sains + sedikit sosial (arah kesehatan/psikologi)
KELAS_SAINS_UMUM  = ["F9", "F10"]   # sains umum

def skor_relevansi_jurusan(rumpun, j_text, kelas, siswa):
    # Skor relevansi SATU jurusan terhadap satu kelas, BERGRADASI 1 sampai 5.
    #   5 = sangat relevan, 4 = relevan, 3 = netral, 2 = kurang relevan, 1 = tidak relevan.
    # Dihitung terpisah untuk Jurusan 1 dan Jurusan 2 agar bobotnya tidak digabung.
    # Relevansi murni soal kecocokan rumpun jurusan dengan profil kelas; nilai akademik
    # siswa sudah ditangani kriteria c1 sampai c3, jadi tidak diikutkan di sini.  
    if rumpun in ["SOSIAL", "TERAPAN_SOSIAL", "HUMANIORA", "AGAMA"]:
        if kelas == "F2": return 5      # paling sosial
        if kelas == "F1": return 4      # sosial
        if kelas in KELAS_SAINS_KES: return 3
        return 1                         # kelas sains murni: tidak relevan

    if rumpun == "FORMAL":  # Matematika, Statistika, Ilmu Komputer
        if kelas in KELAS_SAINS_MAT: return 5
        if kelas in KELAS_SAINS_TEK: return 4
        if kelas in KELAS_SAINS_UMUM: return 3
        return 2

    if rumpun == "ALAM":  # Fisika, Kimia, Biologi -> diarahkan ke F3, F4, F5, F6
        if kelas in KELAS_SAINS_MAT: return 5
        if kelas in KELAS_SAINS_TEK: return 5
        if kelas in KELAS_SAINS_KES: return 3
        if kelas in KELAS_SAINS_UMUM: return 3
        return 1

    if rumpun == "TERAPAN_TEKNIK":  # Teknik, Arsitektur, Sipil, dsb
        if kelas in KELAS_SAINS_TEK: return 5
        if kelas in KELAS_SAINS_MAT: return 4
        if kelas in KELAS_SAINS_UMUM: return 3
        return 2

    if rumpun == "TERAPAN_MEDIS":  # Kedokteran, Psikologi -> diarahkan ke F7, F8
        if kelas in KELAS_SAINS_KES and siswa["Rata_Alam"] >= 80: return 5
        if kelas in KELAS_SAINS_MAT: return 3
        if kelas in KELAS_SAINS_TEK: return 3
        if kelas in KELAS_SAINS_UMUM: return 4
        return 1

    if rumpun == "TERAPAN_AGRO":  # Pertanian, Peternakan -> diarahkan ke F9, F10
        if kelas in KELAS_SAINS_UMUM: return 5
        if kelas in (KELAS_SAINS_MAT + KELAS_SAINS_TEK + KELAS_SAINS_KES): return 3
        return 1

    if rumpun == "KEDINASAN":
        if any(k in j_text for k in ["STAN", "STIS", "AKMIL"]):
            return 5 if kelas in KELAS_SAINS_MAT else 2
        if "STMKG" in j_text:
            return 5 if kelas in KELAS_SAINS_TEK else 2
        if any(k in j_text for k in ["IPDN", "POLTEKIP", "STTD"]):
            return 5 if kelas in KELAS_SOSIAL else 2
        return 3  # kedinasan lain yang belum dipetakan: netral

    return 3  # UMUM atau tidak terklasifikasi: netral

# Label kriteria untuk alasan (urutan sama dengan kolom matriks TOPSIS)
LABEL_KRITERIA_ALASAN = {
    0: "Nilai Alam", 1: "Nilai Sosial", 2: "Nilai Matematika",
    3: "Psikotes", 4: "Minat & Orang Tua",
    5: "Relevansi Jurusan Kuliah 1", 6: "Relevansi Jurusan Kuliah 2",
}

def hasilkan_alasan(v_terpilih, A_pos_global, A_neg_global, closeness):
    # Jarak dihitung berdasarkan nilai vektor siswa di kelas tersebut 
    # terhadap Solusi Ideal POSITIF dan NEGATIF GLOBAL
    kontribusi = (v_terpilih - A_neg_global) ** 2 - (A_pos_global - v_terpilih) ** 2
    urut = np.argsort(kontribusi)[::-1]
    penentu = [LABEL_KRITERIA_ALASAN[int(j)] for j in urut[:3] if kontribusi[j] > 0]
    if not penentu:
        penentu = [LABEL_KRITERIA_ALASAN[int(urut[0])]]
    return f"Closeness TOPSIS Global {closeness:.3f}. Faktor penentu: " + ", ".join(penentu) + "."


# validasi
def validasi_unggahan(df):
    # Periksa kelengkapan kolom dan tipe data sebelum kalkulasi.
    masalah = []
    hilang = [c for c in KOLOM_UPLOAD if c not in df.columns]
    if hilang:
        masalah.append("Kolom wajib tidak ditemukan: " + ", ".join(hilang) + ".")
    for c in KOLOM_NUMERIK:
        if c in df.columns and pd.to_numeric(df[c], errors='coerce').isna().any():
            masalah.append(f"Kolom {c} memuat nilai yang bukan angka.")
    if "NIS" in df.columns and df["NIS"].astype(str).duplicated().any():
        masalah.append("Terdapat NIS ganda pada berkas. NIS harus unik.")
    return masalah

def validasi_penempatan(df_semua):
    # Periksa kuota dan syarat matematika atas keseluruhan penempatan.
    masalah = []
    hitung = df_semua['Verifikasi_Guru'].value_counts()
    for k, kuota in KUOTA_KELAS.items():
        n = int(hitung.get(k, 0))
        if n > kuota:
            masalah.append(f"Kelas {k} melebihi kuota ({n}/{kuota}).")
    for _, r in df_semua.iterrows():
        if r['Verifikasi_Guru'] in KELAS_MAT_LANJUT:
            try:
                if float(r['Rata_Matematika']) < AMBANG_MATEMATIKA_MAT_LANJUT:
                    masalah.append(
                        f"{r['Nama']} (Matematika {r['Rata_Matematika']}) tidak memenuhi "
                        f"syarat minimal {AMBANG_MATEMATIKA_MAT_LANJUT} untuk kelas {r['Verifikasi_Guru']}."
                    )
            except (TypeError, ValueError):
                pass
    return masalah


# proses SPK (AHP-TOPSIS + PENUGASAN OPTIMAL / HUNGARIAN) — MODE GLOBAL
def proses_spk(df, bobot=None):
    if bobot is None:
        bobot, _, _ = hitung_bobot_ahp(matriks_dari_bobot(BOBOT_DEFAULT_PERSEN))

    # Urutkan siswa secara kanonik berdasarkan NIS agar hasil selalu konsisten
    df = df.sort_values(by="NIS", key=lambda s: s.astype(str)).reset_index(drop=True)

    list_kelas = list(PAKET_KELAS.keys())
    idx_kelas = {k: j for j, k in enumerate(list_kelas)}
    n_siswa = len(df)
    n_kelas = len(list_kelas)

    # Matriks Keputusan Raksasa: baris = (n_siswa * 10 kelas), kolom = 7 kriteria
    matriks_keputusan_global = []
    # Mapping untuk mencatat baris ini milik siswa siapa dan kelas apa
    mapping_kombinasi = [] 

    # --- Tahap 1: Bangun Nilai Kriteria untuk Semua Kombinasi ---
    for i, siswa in df.iterrows():
        jur1, jur2 = siswa.get("KULIAH JURUSAN 1", ""), siswa.get("KULIAH JURUSAN 2", "")
        rumpun1, rumpun2 = dapatkan_rumpun_ilmu(jur1), dapatkan_rumpun_ilmu(jur2)

        for j, kelas in enumerate(list_kelas):
            profil = PAKET_KELAS[kelas]
            
            # c1, c2, c3: Nilai Akademik
            c1 = siswa["Rata_Alam"] * (profil["Alam"] / 5)
            c2 = siswa["Rata_Sosial"] * (profil["Sosial"] / 5)
            c3 = siswa["Rata_Matematika"] * (profil["Matematika"] / 5)

            # c4: Psikotes (cabang terakhir = default aman bila profil kelas seimbang)
            if profil["Alam"] > profil["Sosial"]:
                c4 = 5 if siswa["Psikotes"] == "IPA" else 1
            elif profil["Sosial"] > profil["Alam"]:
                c4 = 5 if siswa["Psikotes"] == "IPS" else 1
            else:
                c4 = 3

            # c5: Minat Siswa + Ortu
            c5 = 1
            if siswa["Minat"] == "IPA" and profil["Alam"] >= 4: c5 += 2
            if siswa["Minat"] == "IPS" and profil["Sosial"] >= 4: c5 += 2

            # c6, c7: Relevansi Jurusan (bergradasi 1 sampai 5)
            c6 = skor_relevansi_jurusan(rumpun1, str(jur1).upper(), kelas, siswa)
            c7 = skor_relevansi_jurusan(rumpun2, str(jur2).upper(), kelas, siswa)

            matriks_keputusan_global.append([c1, c2, c3, c4, c5, c6, c7])
            mapping_kombinasi.append((i, j))

    X_global = np.array(matriks_keputusan_global)

    # --- Tahap 2: Normalisasi Vektor Tingkat Global ---
    sum_sq = np.sqrt(np.sum(X_global ** 2, axis=0))
    norm_X_global = np.zeros_like(X_global, dtype=float)
    for col in range(X_global.shape[1]):
        if sum_sq[col] > 0:
            norm_X_global[:, col] = X_global[:, col] / sum_sq[col]

    # --- Tahap 3: Pembobotan Global ---
    V_global = norm_X_global * bobot

    # --- Tahap 4: Tentukan Batas Ideal Terbaik & Terburuk se-Angkatan ---
    A_pos = np.max(V_global, axis=0)
    A_neg = np.min(V_global, axis=0)

    # --- Tahap 5: Hitung Jarak Jauh-Dekat & Closeness Global ---
    S_pos = np.sqrt(np.sum((V_global - A_pos) ** 2, axis=1))
    S_neg = np.sqrt(np.sum((V_global - A_neg) ** 2, axis=1))
    C_global = S_neg / (S_pos + S_neg + 1e-9)

    # --- Tahap 6: Kembalikan Nilai ke Matriks Asli (Siswa x Kelas) ---
    C_mat = np.zeros((n_siswa, n_kelas))
    V_terpetakan = np.zeros((n_siswa, n_kelas, N_KRITERIA)) # Untuk tracking alasan keputusan

    for idx, (i, j) in enumerate(mapping_kombinasi):
        C_mat[i, j] = C_global[idx]
        V_terpetakan[i, j] = V_global[idx]

    # --- Tahap 7: Optimasi Alokasi Kursi (Hungarian Assignment) ---
    nilai_mat = df["Rata_Matematika"].astype(float).to_numpy()
    sisa_kuota = {k: KUOTA_KELAS[k] for k in list_kelas}

    slot_kelas = []
    for k in list_kelas:
        slot_kelas.extend([k] * sisa_kuota[k])
    n_slot = len(slot_kelas)

    PENALTI_TERLARANG = 1000.0   
    BIAYA_TANPA_KELAS = 500.0    

    biaya_kelas = {}
    for k in list_kelas:
        kolom = 1.0 - C_mat[:, idx_kelas[k]]
        if k in KELAS_MAT_LANJUT:
            kolom = np.where(nilai_mat < AMBANG_MATEMATIKA_MAT_LANJUT, PENALTI_TERLARANG, kolom)
        biaya_kelas[k] = kolom

    n_dummy = max(0, n_siswa - n_slot)
    n_kolom = n_slot + n_dummy
    biaya = np.full((n_siswa, n_kolom), BIAYA_TANPA_KELAS, dtype=float)
    for s, k in enumerate(slot_kelas):
        biaya[:, s] = biaya_kelas[k]

    baris_idx, kolom_idx = linear_sum_assignment(biaya)

    rekomendasi = [None] * n_siswa
    alasan = [None] * n_siswa
    for i, s in zip(baris_idx, kolom_idx):
        if s < n_slot and biaya[i, s] < PENALTI_TERLARANG:
            kelas = slot_kelas[s]
            rekomendasi[i] = kelas
            alasan[i] = hasilkan_alasan(
                V_terpetakan[i, idx_kelas[kelas]], 
                A_pos, 
                A_neg, 
                C_mat[i, idx_kelas[kelas]]
            )

    df_hasil = df.copy()
    df_hasil['Rekomendasi_Sistem'] = rekomendasi
    df_hasil['Verifikasi_Guru'] = rekomendasi
    df_hasil['Alasan_Rekomendasi'] = alasan

    return df_hasil


# query database
def simpan_ke_db(df):
    conn = sqlite3.connect('penjurusan.db')
    c = conn.cursor()
    for _, row in df.iterrows():
        c.execute('''
            INSERT OR REPLACE INTO hasil_penjurusan
            (NIS, Nama, Kelas_Lama, Psikotes, Minat,Rata_Alam, Rata_Sosial, Rata_Matematika,
             Jurusan_1, Jurusan_2, Rekomendasi_Sistem, Alasan_Rekomendasi, Verifikasi_Guru)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(row['NIS']), row['Nama'], row['Kelas_Lama'], row['Psikotes'], row['Minat'],
              row['Rata_Alam'], row['Rata_Sosial'], row['Rata_Matematika'],
              str(row.get('KULIAH JURUSAN 1', '')), str(row.get('KULIAH JURUSAN 2', '')),
              row['Rekomendasi_Sistem'], row['Alasan_Rekomendasi'], row['Verifikasi_Guru']))
    conn.commit()
    conn.close()

def load_dari_db():
    conn = sqlite3.connect('penjurusan.db')
    df = pd.read_sql_query("SELECT * FROM hasil_penjurusan", conn)
    conn.close()
    return df

def kosongkan_db():
    # Mode global: hasil = satu kalkulasi utuh, jadi database diganti seluruhnya.
    conn = sqlite3.connect('penjurusan.db')
    c = conn.cursor()
    c.execute("DELETE FROM hasil_penjurusan")
    conn.commit()
    conn.close()

def hapus_kelas_dari_db(kelas_lama):
    conn = sqlite3.connect('penjurusan.db')
    c = conn.cursor()
    c.execute("DELETE FROM hasil_penjurusan WHERE Kelas_Lama = ?", (kelas_lama,))
    conn.commit()
    conn.close()

def update_kelas_db(df_edited):
    conn = sqlite3.connect('penjurusan.db')
    c = conn.cursor()
    for _, row in df_edited.iterrows():
        c.execute("UPDATE hasil_penjurusan SET Verifikasi_Guru = ? WHERE NIS = ?", (row['Verifikasi_Guru'], str(row['NIS'])))
    conn.commit()
    conn.close()


# data dummy
def generate_mock_data():
    np.random.seed()
    data = {
        "NIS": [f"{np.random.randint(10000, 99999)}" for _ in range(30)],
        "Nama": [f"Siswa Simulasi {i}" for i in range(1, 31)],
        "Psikotes": np.random.choice(["IPA", "IPS"], 30),
        "Minat": np.random.choice(["IPA", "IPS"], 30),
        "Rata_Alam": np.random.uniform(72, 96, 30).round(1),
        "Rata_Sosial": np.random.uniform(70, 95, 30).round(1),
        "Rata_Matematika": np.random.uniform(75, 94, 30).round(1),
        "KULIAH JURUSAN 1": np.random.choice(
            ["Teknik Informatika", "Sosiologi", "STMKG", "STIS", "Peternakan", "Hukum", "Kedokteran"], 30),
        "KULIAH JURUSAN 2": np.random.choice(
            ["STAN", "Pertanian", "Fisika", "IPDN", "Akpol", "Manajemen"], 30)
    }
    return pd.DataFrame(data)

# export excel
def generate_excel_bytes(df, group_by_col):
    output = io.BytesIO()
    kolom_excel = [
        "NIS", "Nama", "Kelas_Lama", "Verifikasi_Guru", 
        "Psikotes", "Minat", "Rata_Alam", "Rata_Sosial", "Rata_Matematika", 
        "Jurusan_1", "Jurusan_2", "Alasan_Rekomendasi"
    ]
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for nama_grup, df_grup in df.groupby(group_by_col):
            df_grup = df_grup.sort_values(by="Nama")
            df_grup[kolom_excel].rename(columns={
                "Verifikasi_Guru": "Kelas_Baru",
                "Jurusan_1": "KULIAH JURUSAN 1",
                "Jurusan_2": "KULIAH JURUSAN 2"
            }).to_excel(writer, sheet_name=str(nama_grup), index=False)
    return output.getvalue()


#kode desain
def muat_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg-main: #F8FAFC;       --bg-panel: #FFFFFF;      --text-primary: #0F172A;
        --text-muted: #64748B;    --brand-primary: #4F46E5; --brand-hover: #4338CA;
        --border-color: #E2E8F0;  --radius-md: 8px;         --radius-lg: 12px;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; color: var(--text-primary); }
    .stApp { background-color: var(--bg-main); }
    h1, h2, h3, h4, h5, h6 { font-weight: 600; color: var(--text-primary); letter-spacing: -0.02em; }
    [data-testid="stSidebar"] { background-color: var(--bg-panel); border-right: 1px solid var(--border-color); }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-color); margin-bottom: 1.5rem; }
    div[data-testid="stMetric"] {
        background-color: var(--bg-panel); border: 1px solid var(--border-color); border-radius: var(--radius-lg);
        padding: 1.25rem; box-shadow: var(--shadow-sm); transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetricValue"] { font-weight: 700; font-size: 2rem; color: var(--text-primary); margin-top: 0.2rem; }
    .stButton button[kind="primary"] { background-color: var(--brand-primary); color: white; border: none; border-radius: var(--radius-md); font-weight: 500; padding: 0.5rem 1.2rem; box-shadow: var(--shadow-sm); }
    .stButton button[kind="primary"]:hover { background-color: var(--brand-hover); box-shadow: var(--shadow-md); }
    .stButton button:not([kind="primary"]) { background-color: var(--bg-panel); border: 1px solid var(--border-color); color: var(--text-primary); border-radius: var(--radius-md); font-weight: 500; }
    .stButton button:not([kind="primary"]):hover { border-color: var(--brand-primary); color: var(--brand-primary); }
    .stDownloadButton button { background-color: #EEF2FF; border: 1px solid #C7D2FE; color: var(--brand-primary); border-radius: var(--radius-md); font-weight: 600; }
    .stDownloadButton button:hover { background-color: #E0E7FF; border-color: var(--brand-primary); }
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] { background-color: var(--bg-panel); border: 1px solid var(--border-color); border-radius: var(--radius-lg); box-shadow: var(--shadow-sm); padding: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--border-color); }
    .stTabs [data-baseweb="tab"] { font-weight: 500; color: var(--text-muted); padding-top: 1rem; padding-bottom: 1rem; }
    .stTabs [aria-selected="true"] { color: var(--brand-primary) !important; border-bottom-color: var(--brand-primary) !important; }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem; font-weight: 700; color: var(--brand-primary); margin-bottom: 0.25rem; display: inline-block; background-color: #EEF2FF; padding: 0.2rem 0.6rem; border-radius: 4px; }
    div[data-testid="stAlert"] { border-radius: var(--radius-md); border: 1px solid var(--border-color); }
    hr { border-color: var(--border-color) !important; margin: 2rem 0 !important; }
    </style>
    """, unsafe_allow_html=True)


def badge_kuota(terisi, total):
    if total == 0: return "—"
    rasio = terisi / total
    if rasio >= 1: return "Penuh"
    elif rasio >= 0.8: return "Hampir penuh"
    else: return "Tersedia"


# isi streamlit
muat_css()

# State mode global
if 'data_terkumpul' not in st.session_state: st.session_state.data_terkumpul = {}   # {kelas_asal: df}
if 'hasil_global' not in st.session_state: st.session_state.hasil_global = None
if 'bobot_ahp' not in st.session_state:
    st.session_state.bobot_ahp, _, _ = hitung_bobot_ahp(matriks_dari_bobot(BOBOT_DEFAULT_PERSEN))

with st.sidebar:
    st.markdown("<h1>SPK Penempatan Kelas SMAN 7 BEKASI</h1>", unsafe_allow_html=True)
    st.caption("Sistem Pendukung Keputusan penempatan kelas F1–F10")

    st.write("")
    halaman = st.radio(
        "Menu Navigasi",
        ["Unggah & Kalkulasi", "Database & Manajemen", "Pengaturan Bobot %"],
        label_visibility="collapsed"
    )


# HALAMAN 1: KUMPULKAN DATA SEMUA KELAS, LALU KALKULASI GLOBAL
if halaman == "Unggah & Kalkulasi":
    st.markdown('<div class="eyebrow">STEP 1 / 3</div>', unsafe_allow_html=True)
    st.title("Kumpulkan Data & Jalankan Kalkulasi")
    st.caption(
        "Unggah data tiap kelas asal lebih dulu sampai semua kelas masuk daftar. "
        "Setelah lengkap, jalankan kalkulasi satu kali untuk seluruh siswa. "
        "Karena semua siswa dihitung sekaligus, hasil penempatan tidak lagi bergantung pada urutan unggah."
    )

    # ---------- 1. Tambah data per kelas ----------
    st.write("##### 1. Tambah Data per Kelas")
    pilihan_kelas = [f"X{i}" for i in range(1, 13)]
    kelas_dipilih = st.selectbox(
        "Kelas asal yang datanya akan ditambahkan",
        ["-- Pilih Kelas --"] + pilihan_kelas,
        label_visibility="collapsed"
    )

    if kelas_dipilih != "-- Pilih Kelas --":
        if kelas_dipilih in st.session_state.data_terkumpul:
            st.info(
                f"{kelas_dipilih} sudah ada di daftar "
                f"({len(st.session_state.data_terkumpul[kelas_dipilih])} siswa). "
                "Menambahkan lagi akan menimpa data lamanya."
            )
        st.info(f"Pastikan kolom ini tersedia: {', '.join(KOLOM_UPLOAD)}", icon="ℹ️")

        col_up, col_mock = st.columns([3, 1])
        with col_up:
            file_unggahan = st.file_uploader(
                f"File CSV/Excel untuk {kelas_dipilih}", type=['csv', 'xlsx'],
                label_visibility="collapsed", key=f"up_{kelas_dipilih}"
            )
        with col_mock:
            if st.button("Pakai Data Dummy", use_container_width=True, key=f"dum_{kelas_dipilih}"):
                df_dummy = generate_mock_data()
                df_dummy["Kelas_Lama"] = kelas_dipilih
                st.session_state.data_terkumpul[kelas_dipilih] = df_dummy
                st.session_state.hasil_global = None
                st.success(f"30 data dummy ditambahkan untuk {kelas_dipilih}.")
                st.rerun()

        if file_unggahan is not None:
            try:
                if file_unggahan.name.endswith('.csv'):
                    df_upload = pd.read_csv(file_unggahan)
                else:
                    df_upload = pd.read_excel(file_unggahan)

                masalah_unggah = validasi_unggahan(df_upload)
                if masalah_unggah:
                    st.error("Berkas tidak dapat diproses. Perbaiki dulu:")
                    for m in masalah_unggah:
                        st.write("• " + m)
                else:
                    for c in KOLOM_NUMERIK:
                        df_upload[c] = pd.to_numeric(df_upload[c], errors='coerce')
                    df_upload["Kelas_Lama"] = kelas_dipilih
                    st.write("Pratinjau:")
                    st.dataframe(df_upload[KOLOM_UPLOAD].head(), use_container_width=True, hide_index=True)
                    if st.button(f"➕ Tambahkan {kelas_dipilih} ke Daftar", type="primary", key=f"add_{kelas_dipilih}"):
                        st.session_state.data_terkumpul[kelas_dipilih] = df_upload
                        st.session_state.hasil_global = None
                        st.success(f"{kelas_dipilih} ditambahkan ke daftar ({len(df_upload)} siswa).")
                        st.rerun()
            except Exception as e:
                st.error(f"Gagal memuat file: {e}")

    st.divider()

    # ---------- 2. Daftar kelas siap dihitung ----------
    st.write("##### 2. Daftar Kelas Siap Dihitung")
    if not st.session_state.data_terkumpul:
        st.warning("Belum ada kelas di daftar. Tambahkan minimal satu kelas di atas.")
    else:
        ringkas = pd.DataFrame(
            [{"Kelas Asal": k, "Jumlah Siswa": len(v)} for k, v in st.session_state.data_terkumpul.items()]
        )
        total_siswa = int(ringkas["Jumlah Siswa"].sum())
        total_kuota = sum(KUOTA_KELAS.values())

        c1, c2, c3 = st.columns(3)
        c1.metric("Kelas di daftar", len(ringkas))
        c2.metric("Total siswa", total_siswa)
        c3.metric("Total kuota kelas baru", total_kuota)
        st.dataframe(ringkas, use_container_width=True, hide_index=True)

        col_hapus, col_reset = st.columns([2, 1])
        with col_hapus:
            kelas_hapus = st.selectbox(
                "Hapus satu kelas dari daftar",
                ["-- Pilih --"] + list(st.session_state.data_terkumpul.keys())
            )
            if kelas_hapus != "-- Pilih --":
                if st.button(f"Hapus {kelas_hapus} dari daftar"):
                    st.session_state.data_terkumpul.pop(kelas_hapus, None)
                    st.session_state.hasil_global = None
                    st.rerun()
        with col_reset:
            st.write("")
            st.write("")
            if st.button("Kosongkan Seluruh Daftar"):
                st.session_state.data_terkumpul = {}
                st.session_state.hasil_global = None
                st.rerun()

        if total_siswa > total_kuota:
            st.warning(
                f"Total siswa ({total_siswa}) melebihi total kuota ({total_kuota}). "
                "Sebagian siswa akan berstatus tanpa kelas.",
                icon="⚠️"
            )

        bobot_terkini = np.asarray(st.session_state.bobot_ahp, dtype=float)
        st.info(
            "Bobot aktif (%): " +
            " | ".join([f"**{n}** {w*100:.1f}" for n, w in zip(NAMA_KRITERIA_AHP, bobot_terkini)])
        )

        # ---------- 3. Jalankan kalkulasi global ----------
        st.write("##### 3. Jalankan Kalkulasi Global")
        st.caption(
            "Kalkulasi memproses SEMUA siswa di daftar sekaligus dalam satu penugasan optimal "
            "terhadap kuota penuh. Pastikan semua kelas sudah masuk daftar sebelum menjalankan."
        )
        if st.button("🚀 Jalankan Kalkulasi untuk Semua Kelas", type="primary"):
            df_all = pd.concat(list(st.session_state.data_terkumpul.values()), ignore_index=True)
            if df_all["NIS"].astype(str).duplicated().any():
                dups = df_all.loc[df_all["NIS"].astype(str).duplicated(keep=False), "NIS"].unique()
                st.error("Terdapat NIS ganda antar kelas: " + ", ".join(map(str, dups)) + ". Perbaiki dulu.")
            else:
                with st.spinner('Menghitung seluruh siswa (AHP-TOPSIS + penugasan optimal global)...'):
                    st.session_state.hasil_global = proses_spk(df_all, bobot=bobot_terkini)
                tanpa_kelas = st.session_state.hasil_global['Rekomendasi_Sistem'].isna().sum()
                if tanpa_kelas > 0:
                    st.warning(
                        f"{tanpa_kelas} siswa belum mendapat kelas karena kuota tidak cukup "
                        f"atau terkendala syarat matematika F3/F4. Tinjau manual di tabel bawah.",
                        icon="⚠️"
                    )
                st.success("Kalkulasi global selesai. Penempatan optimal untuk seluruh siswa.")

    # ---------- 4. Verifikasi guru & simpan ----------
    if st.session_state.hasil_global is not None:
        st.divider()
        st.markdown('<div class="eyebrow">STEP 2 / 3</div>', unsafe_allow_html=True)
        st.write("### Verifikasi Guru (Seluruh Siswa)")
        st.caption(
            "Ubah nilai pada kolom **Keputusan Final** apabila Anda tidak setuju dengan hasil algoritma. "
            "Sistem akan menolak penempatan yang melanggar kuota atau syarat matematika."
        )

        config_kolom = {
            "Verifikasi_Guru": st.column_config.SelectboxColumn(
                "Keputusan Final (Klik ⬇)", options=list(PAKET_KELAS.keys()), required=True),
            "Rekomendasi_Sistem": st.column_config.TextColumn("Rekomendasi SPK", disabled=True),
            "Alasan_Rekomendasi": st.column_config.TextColumn("Alasan (TOPSIS)", disabled=True),
        }
        for col in KOLOM_UPLOAD + ["Kelas_Lama"]:
            config_kolom[col] = st.column_config.TextColumn(disabled=True)

        df_edit = st.data_editor(
            st.session_state.hasil_global[KOLOM_EDITOR],
            column_config=config_kolom, use_container_width=True, hide_index=True, height=480,
            key="editor_global"
        )

        st.divider()
        st.markdown('<div class="eyebrow">STEP 3 / 3</div>', unsafe_allow_html=True)
        st.caption(
            "Karena ini hasil satu kalkulasi global yang utuh, menyimpan akan mengganti seluruh isi "
            "database dengan hasil ini. Pastikan semua kelas sudah ikut dihitung."
        )
        konfirmasi = st.checkbox("Saya paham menyimpan akan mengganti seluruh isi database dengan hasil ini.")
        if st.button("Sahkan & Simpan ke Database Induk", type="primary", disabled=not konfirmasi):
            df_final = st.session_state.hasil_global.copy()
            df_final['Verifikasi_Guru'] = df_edit['Verifikasi_Guru'].values

            masalah = validasi_penempatan(df_final[['NIS', 'Nama', 'Rata_Matematika', 'Verifikasi_Guru']])
            if masalah:
                st.error("Penempatan ditolak. Perbaiki dulu kolom Keputusan Final:")
                for m in masalah:
                    st.write("• " + m)
            else:
                kosongkan_db()
                simpan_ke_db(df_final)
                st.session_state.data_terkumpul = {}
                st.session_state.hasil_global = None
                
                st.success("Seluruh data berhasil disahkan dan disimpan ke database!")
                pesan_tunggu = st.empty() 
                
                for detik in range(5, 0, -1):
                    pesan_tunggu.info(f"Halaman akan dimuat ulang dalam {detik} detik...")
                    time.sleep(1)
                pesan_tunggu.empty()
                st.rerun()

# HALAMAN 2: HASIL AKHIR & MANAJEMEN DATA
elif halaman == "Database & Manajemen":
    st.markdown('<div class="eyebrow">DASHBOARD</div>', unsafe_allow_html=True)
    st.title("Ringkasan & Distribusi Kelas")

    df_db = load_dari_db()

    if df_db.empty:
        st.warning("Database masih kosong. Belum ada hasil yang disahkan.")
    else:
        st.write("##### Sisa Kuota Global")
        cols = st.columns(5)
        for i, (k, kuota_max) in enumerate(KUOTA_KELAS.items()):
            terisi = len(df_db[df_db['Verifikasi_Guru'] == k])
            sisa = kuota_max - terisi
            status = badge_kuota(terisi, kuota_max)
            with cols[i % 5]:
                st.metric(label=f"Kelas {k}", value=f"{terisi} / {kuota_max}", delta=f"Sisa: {sisa} ({status})",
                          delta_color="off" if sisa <= 0 else "normal")

        st.divider()

        tab1, tab2, tab_admin = st.tabs(
            ["📊 Data Kelas Baru", "🗂️ Data Kelas Asal", "⚙️ Manajemen Database"])
        
        kolom_db_tampil = [
            "NIS", "Nama", "Kelas_Lama", "Verifikasi_Guru", 
            "Psikotes", "Minat", "Rata_Alam", "Rata_Sosial", "Rata_Matematika", 
            "Jurusan_1", "Jurusan_2", "Alasan_Rekomendasi"
        ]

        config_db = {
            "Verifikasi_Guru": st.column_config.SelectboxColumn("Kelas Baru (Bisa Diedit ⬇)", options=list(PAKET_KELAS.keys()), required=True),
            "Jurusan_1": st.column_config.TextColumn("KULIAH JURUSAN 1", disabled=True),
            "Jurusan_2": st.column_config.TextColumn("KULIAH JURUSAN 2", disabled=True),
            "Alasan_Rekomendasi": st.column_config.TextColumn("Alasan (TOPSIS)", disabled=True)
        }
        for col in kolom_db_tampil:
            if col not in config_db:
                config_db[col] = st.column_config.TextColumn(disabled=True)

        def simpan_dengan_validasi(edited_df, df_basis, kunci):
            # Terapkan suntingan ke salinan seluruh data, lalu validasi kuota & syarat.
            df_cek = df_basis.copy().set_index(df_basis['NIS'].astype(str))
            for _, row in edited_df.iterrows():
                df_cek.loc[str(row['NIS']), 'Verifikasi_Guru'] = row['Verifikasi_Guru']
            df_cek = df_cek.reset_index(drop=True)
            masalah = validasi_penempatan(df_cek)
            if masalah:
                st.error("Perubahan ditolak. Perbaiki dulu:")
                for m in masalah:
                    st.write("• " + m)
                return False
            update_kelas_db(edited_df)
            return True

        with tab1:
            col_a, col_b = st.columns([1, 3])
            with col_a:
                pilihan_kelas_baru = st.selectbox("Filter berdasarkan Kelas Baru:", list(PAKET_KELAS.keys()))
            
            st.caption("Ubah status pada kolom **Kelas Baru** jika ingin memindahkan siswa. Sistem menolak pemindahan yang melanggar kuota atau syarat matematika.")
            df_filtered_1 = df_db[df_db["Verifikasi_Guru"] == pilihan_kelas_baru][kolom_db_tampil]
            
            edited_df1 = st.data_editor(
                df_filtered_1,
                column_config=config_db, use_container_width=True, hide_index=True, key="edit_tab1"
            )

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("💾 Simpan Perubahan Kelas", key="btn_save1", type="primary"):
                    if simpan_dengan_validasi(edited_df1, df_db, "tab1"):
                        st.success("Perubahan data kelas siswa berhasil disimpan!")
                        st.rerun()

            st.download_button(
                label="⬇ Unduh Excel (Seluruh Kelas Baru)",
                data=generate_excel_bytes(df_db, group_by_col="Verifikasi_Guru"),
                file_name="Rekap_Kelas_Baru_F1_F10.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with tab2:
            list_kelas_lama = sorted(df_db["Kelas_Lama"].unique())
            col_x, col_y = st.columns([1, 3])
            with col_x:
                pilihan_kelas_lama = st.selectbox("Filter berdasarkan Kelas Asal:", list_kelas_lama, key="sel_lama")
                
            st.caption("Ubah status pada kolom **Kelas Baru** jika ingin memindahkan siswa. Sistem menolak pemindahan yang melanggar kuota atau syarat matematika.")
            df_filtered_2 = df_db[df_db["Kelas_Lama"] == pilihan_kelas_lama][kolom_db_tampil]
            
            edited_df2 = st.data_editor(
                df_filtered_2,
                column_config=config_db, use_container_width=True, hide_index=True, key="edit_tab2"
            )

            col_btn2, _ = st.columns([1, 3])
            with col_btn2:
                if st.button("💾 Simpan Perubahan Kelas", key="btn_save2", type="primary"):
                    if simpan_dengan_validasi(edited_df2, df_db, "tab2"):
                        st.success("Perubahan data kelas siswa berhasil disimpan!")
                        st.rerun()

            st.download_button(
                label="⬇ Unduh Excel (Berdasarkan Kelas Asal)",
                data=generate_excel_bytes(df_db, group_by_col="Kelas_Lama"),
                file_name="Distribusi_Dari_Kelas_Lama.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with tab_admin:
            st.write("##### Hapus Data Kelas")
            st.info(
                "Fitur ini digunakan jika terjadi kesalahan unggah atau Anda perlu menghitung ulang suatu kelas dari awal.",
                icon="💡"
            )

            kelas_tersedia_di_db = sorted(df_db["Kelas_Lama"].unique())
            col_del, _ = st.columns([2, 2])
            with col_del:
                kelas_yg_dihapus = st.selectbox(
                    "Pilih kelas yang akan dihapus permanen",
                    ["-- Pilih --"] + list(kelas_tersedia_di_db)
                )

                if kelas_yg_dihapus != "-- Pilih --":
                    if st.button(f"Hapus Semua Data {kelas_yg_dihapus}", type="primary", use_container_width=True):
                        hapus_kelas_dari_db(kelas_yg_dihapus)
                        st.success(f"Data {kelas_yg_dihapus} berhasil dihapus. Kuota telah dikembalikan.")
                        st.rerun()

# HALAMAN 3: PENGATURAN BOBOT (INPUT PERSEN, MATRIKS SAATY MENGIKUTI)
elif halaman == "Pengaturan Bobot %":
    st.markdown('<div class="eyebrow">KONFIGURASI</div>', unsafe_allow_html=True)
    st.title("Pengaturan Bobot Kriteria")
    st.caption(
        "Atur langsung bobot kepentingan tiap kriteria dalam persen. "
        "Pastikan total seluruh bobot adalah tepat 100%."
    )

    bobot_sekarang = np.asarray(st.session_state.bobot_ahp, dtype=float)
    persen_default = bobot_sekarang * 100

    st.write("##### Bobot Kepentingan (%)")
    
    nilai_input = []
    cols = st.columns(3)
    for i, nama in enumerate(NAMA_KRITERIA_AHP):
        with cols[i % 3]:
            # Ubah min_value ke 0.0 agar user lebih fleksibel saat menyeimbangkan angka
            v = st.number_input(
                nama, min_value=0.0, max_value=100.0,
                value=float(round(persen_default[i], 1)), step=1.0, key=f"w_{i}"
            )
            nilai_input.append(v)

    # --- LOGIKA PERHITUNGAN TOTAL & SISA ---
    arr = np.array(nilai_input, dtype=float)
    total_input = round(arr.sum(), 2) # Dibulatkan untuk menghindari bug float
    sisa = round(100.0 - total_input, 2)
    
    # --- MENAMPILKAN INDIKATOR SISA PERSEN ---
    st.write("") # Spasi
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Persentase Saat Ini", f"{total_input}%")
    with col_stat2:
        if sisa == 0:
            st.metric("Sisa Persentase", f"{sisa}%")
        elif sisa > 0:
            st.metric("Sisa Persentase", f"{sisa}%")
        else:
            st.metric("Kelebihan Persentase", f"{abs(sisa)}%")

    # --- PREVIEW AHP ---
    # Normalisasi tetap dilakukan untuk menghindari error pembagian dengan 0 jika total 0
    bobot_preview = arr / arr.sum() if arr.sum() > 0 else arr 
    matrix_preview = matriks_dari_bobot(bobot_preview)
    _, cr_preview, lmax_preview = hitung_bobot_ahp(matrix_preview)

    st.divider()
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        is_disabled = sisa != 0 
        
        if st.button("Terapkan Bobot Ini", type="primary", use_container_width=True, disabled=is_disabled):
            st.session_state.bobot_ahp = bobot_preview
            st.session_state.hasil_global = None
            st.success("Bobot berhasil diterapkan! Kalkulasi berikutnya memakai bobot ini.")
            
    with col_btn2:
        if st.button("Reset ke Default", use_container_width=True):
            st.session_state.bobot_ahp, _, _ = hitung_bobot_ahp(matriks_dari_bobot(BOBOT_DEFAULT_PERSEN))
            st.session_state.hasil_global = None
            for i in range(len(NAMA_KRITERIA_AHP)):
                st.session_state.pop(f"w_{i}", None)
            st.success("Bobot direset ke nilai default.")
            st.rerun()
