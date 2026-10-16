import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
import seaborn as sns
url = "https://raw.githubusercontent.com/vutruong99/CSGO-Round-Winner-Classifier/master/csgo_round_snapshots.csv"

df = pd.read_csv(url)

print(df.shape)
print(df.head())
print(df.select_dtypes(exclude=['int64', 'float64']).columns)
print(df['round_winner'].value_counts())
print(df.isnull().sum().sum())
print(df['map'].value_counts())
print(df['bomb_planted'].value_counts())
df['bomb_planted'] = df['bomb_planted'].astype(int)
df['round_winner'] = df['round_winner'].map({'T':1,'CT':0})
df = pd.get_dummies(df,columns=['map'],drop_first=True,dtype=int)
print("Yeni Boyut:", df.shape)
print(
    "Kalan Metin Sütunları:",
    df.select_dtypes(exclude=['int64', 'float64', 'int32']).columns,
)

y = df['round_winner']
X = df.drop(columns=['round_winner'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)

print("Doğruluk Oranı (Accuracy):", accuracy_score(y_test, y_pred))
print("\nKarmaşıklık Matrisi (Confusion Matrix):\n", confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

katsayilar = pd.Series(model.coef_[0],index=X.columns).sort_values()
print("--- CT Takımına Raundu Kazandıran En Güçlü 5 Faktör (Negatif) ---")
print(katsayilar.head(5))

print("\n--- T Takımına Raundu Kazandıran En Güçlü 5 Faktör (Pozitif) ---")
print(katsayilar.tail(5))

olasiliklar = model.predict_proba(X_test_scaled)
tahmin_tablosu = pd.DataFrame({
    'CT_Kazanma_İhtimali (%)': (olasiliklar[:5, 0] * 100).round(1),
    'T_Kazanma_İhtimali (%)': (olasiliklar[:5, 1] * 100).round(1),
    'Model_Tahmini (0:CT, 1:T)': y_pred[:5],
    'Gercek_Sonuc (0:CT, 1:T)': y_test.iloc[:5].values,
    })

print("\n--- Canlı Yayın Kazanma Olasılığı Simülasyonu (İlk 5 Raund) ---")
print(tahmin_tablosu)


df["health_diff"] = df["ct_health"] - df["t_health"]
df["armor_diff"] = df["ct_armor"] - df["t_armor"]
df["money_diff"] = df["ct_money"] - df["t_money"]
df["alive_diff"] = df["ct_players_alive"] - df["t_players_alive"]

X_yeni = df.drop(columns=["round_winner"])
y_yeni = df["round_winner"]

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_yeni, y_yeni, test_size=0.2, random_state=42
)

scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled = scaler_fe.transform(X_test_fe)

model_fe = LogisticRegression(max_iter=1000)
model_fe.fit(X_train_fe_scaled, y_train_fe)
y_pred_fe = model_fe.predict(X_test_fe_scaled)

print("\n--- Özellik Mühendisliği Sonrası Sonuçlar ---")
print("Yeni Doğruluk Oranı (Accuracy):", accuracy_score(y_test_fe, y_pred_fe))

katsayilar_fe = pd.Series(model_fe.coef_[0], index=X_yeni.columns).sort_values()
print("\n--- Yeni En Güçlü 5 CT Faktörü (Negatif) ---")
print(katsayilar_fe.head(5))
print("\n--- Yeni En Güçlü 5 T Faktörü (Pozitif) ---")
print(katsayilar_fe.tail(5))


analiz_df = X_test[['bomb_planted', 'ct_players_alive', 't_players_alive']].copy()
analiz_df['CT_Kazanma_%'] = (model.predict_proba(X_test_scaled)[:, 0] * 100).round(1)
analiz_df['T_Kazanma_%'] = (model.predict_proba(X_test_scaled)[:, 1] * 100).round(1)
analiz_df['Bomba_Durumu'] = analiz_df['bomb_planted'].map({0: 'Bomba Kurulmadı', 1: 'Bomba Kurulu (C4 Aktif)'})

ozet_tablo = analiz_df.groupby('Bomba_Durumu')[['ct_players_alive', 't_players_alive', 'CT_Kazanma_%', 'T_Kazanma_%']].mean().round(1)
ozet_tablo.columns = ['Ort. Hayattaki CT', 'Ort. Hayattaki T', 'Model CT Kazanma %', 'Model T Kazanma %']
print("\n=== MAÇ İÇİ KRİTİK DURUM ANALİZ TABLOSU ===")
print(ozet_tablo.to_markdown() if hasattr(ozet_tablo, 'to_markdown') else ozet_tablo)



plt.rcdefaults()
plt.style.use('dark_background')

# Senin eğittiğin son modelden (model_fe) T kazanma yüzdelerini alıyoruz
t_olasilik = model_fe.predict_proba(X_test_fe_scaled)[:, 1] * 100

# ------------------------------------------------------------------------------
# GRAFİK 1: 5v5 TAKTİKSEL SAYI ÜSTÜNLÜĞÜ MATRİSİ (1v1 - 5v5)
# ------------------------------------------------------------------------------
taktik_df = pd.DataFrame({
    'CT_Hayatta': X_test_fe['ct_players_alive'].astype(int),
    'T_Hayatta': X_test_fe['t_players_alive'].astype(int),
    'T_Kazanma_Ihtimali': t_olasilik,
})

taktik_df = taktik_df[
    (taktik_df['CT_Hayatta'] >= 1) & (taktik_df['T_Hayatta'] >= 1)
]
pivot_matrix = taktik_df.pivot_table(
    index='CT_Hayatta',
    columns='T_Hayatta',
    values='T_Kazanma_Ihtimali',
    aggfunc='mean',
).sort_index(ascending=False)

fig1, ax1 = plt.subplots(figsize=(9, 7.5))
fig1.patch.set_facecolor('#090d16')
ax1.set_facecolor('#090d16')

cmap_custom = sns.diverging_palette(
    240, 15, s=90, l=55, center='dark', as_cmap=True
)
sns.heatmap(
    pivot_matrix,
    annot=True,
    fmt='.1f',
    cmap=cmap_custom,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'T Kazanma İhtimali (%)'},
    linewidths=2.5,
    linecolor='#090d16',
    annot_kws={'size': 13, 'weight': 'bold', 'color': 'white'},
    ax=ax1,
)

for t in ax1.texts:
  t.set_text(t.get_text() + '%')

ax1.set_title(
    'OYUNCU SAYISINA GÖRE RAUND KAZANMA OLASILIĞI (1v1 - 5v5)',
    fontsize=13,
    fontweight='bold',
    color='white',
    pad=18,
)
ax1.set_xlabel(
    'Hayattaki T Oyuncu Sayısı',
    fontsize=11,
    fontweight='bold',
    color='#fb923c',
    labelpad=10,
)
ax1.set_ylabel(
    'Hayattaki CT Oyuncu Sayısı',
    fontsize=11,
    fontweight='bold',
    color='#38bdf8',
    labelpad=10,
)
ax1.tick_params(axis='both', labelsize=11, labelrotation=0)

plt.savefig(
    '1_cs_tactical_matrix.png',
    dpi=300,
    facecolor=fig1.get_facecolor(),
    bbox_inches='tight',
)
plt.show()

# ------------------------------------------------------------------------------
# GRAFİK 2: OLASILIK YOĞUNLUK DALGASI (%50 CLUTCH BÖLGESİ)
# ------------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
fig2.patch.set_facecolor('#090d16')
ax2.set_facecolor('#111827')

dagilim_df = pd.DataFrame({
    'T_Olasilik': t_olasilik,
    'Gercek_Kazanan': y_test_fe.map(
        {0: 'Gerçekte CT Kazandı', 1: 'Gerçekte T Kazandı'}
    ),
})

sns.kdeplot(
    data=dagilim_df[dagilim_df['Gercek_Kazanan'] == 'Gerçekte CT Kazandı'],
    x='T_Olasilik',
    fill=True,
    color='#38bdf8',
    alpha=0.45,
    linewidth=2.5,
    label='Gerçekte CT Kazanan Raundlar',
    ax=ax2,
)
sns.kdeplot(
    data=dagilim_df[dagilim_df['Gercek_Kazanan'] == 'Gerçekte T Kazandı'],
    x='T_Olasilik',
    fill=True,
    color='#fb923c',
    alpha=0.45,
    linewidth=2.5,
    label='Gerçekte T Kazanan Raundlar',
    ax=ax2,
)

ax2.axvline(50, color='#f43f5e', linestyle='--', linewidth=2, zorder=4)
ax2.text(
    50,
    ax2.get_ylim()[1] * 0.82,
    ' %50 Karar Eşiği\n (Clutch Bölgesi)',
    color='#f43f5e',
    fontweight='bold',
    fontsize=10,
    ha='center',
    bbox=dict(
        boxstyle='round,pad=0.4', facecolor='#1f2937', edgecolor='#f43f5e'
    ),
)

ax2.set_xlim(0, 100)
ax2.set_title(
    'MODELİN SINIF AYRIŞTIRMA GÜCÜ: OLASILIK YOĞUNLUK DALGASI',
    fontsize=13,
    fontweight='bold',
    color='white',
    pad=15,
)
ax2.set_xlabel(
    'Modelin T Takımına Verdiği Kazanma İhtimali (%)',
    fontsize=10,
    color='#cbd5e1',
    labelpad=10,
)
ax2.set_ylabel('Raund Yoğunluğu', fontsize=10, color='#cbd5e1')
ax2.legend(
    loc='upper center',
    frameon=True,
    facecolor='#1f2937',
    edgecolor='none',
    fontsize=10,
)
ax2.grid(True, linestyle=':', alpha=0.25)

plt.savefig(
    '2_cs_probability_wave.png',
    dpi=300,
    facecolor=fig2.get_facecolor(),
    bbox_inches='tight',
)
plt.show()

# ------------------------------------------------------------------------------
# GRAFİK 3: EN GÜÇLÜ 12 FAKTÖR (SENİN KATSAYILAR_FE SERİN İLE LOLLIPOP GRAFİĞİ)
# ------------------------------------------------------------------------------
en_etkili_12 = pd.concat([katsayilar_fe.head(6), katsayilar_fe.tail(6)])

fig3, ax3 = plt.subplots(figsize=(10, 6.5))
fig3.patch.set_facecolor('#090d16')
ax3.set_facecolor('#111827')

renkler = ['#38bdf8' if x < 0 else '#fb923c' for x in en_etkili_12.values]
ax3.hlines(
    y=en_etkili_12.index,
    xmin=0,
    xmax=en_etkili_12.values,
    color=renkler,
    alpha=0.85,
    linewidth=4,
)
ax3.scatter(
    en_etkili_12.values, en_etkili_12.index, color=renkler, s=120, zorder=3
)
ax3.axvline(0, color='#94a3b8', linestyle='--', linewidth=1.2)

for i, (ozellik, deger) in enumerate(en_etkili_12.items()):
  offset = 0.03 if deger > 0 else -0.03
  hizalama = 'left' if deger > 0 else 'right'
  ax3.text(
      deger + offset,
      i,
      f'{deger:.2f}',
      va='center',
      ha=hizalama,
      color='white',
      fontweight='bold',
      fontsize=9.5,
  )

ax3.set_xlim(en_etkili_12.min() - 0.15, en_etkili_12.max() + 0.15)
ax3.set_title(
    'RAUND KADERİNİ BELİRLEYEN EN GÜÇLÜ 12 FAKTÖR (MAVİ: CT, TURUNCU: T)',
    fontsize=12.5,
    fontweight='bold',
    color='white',
    pad=15,
)
ax3.set_xlabel(
    'Lojistik Regresyon Ağırlık Katsayısı',
    fontsize=10,
    color='#cbd5e1',
    labelpad=10,
)
ax3.grid(axis='x', linestyle=':', alpha=0.25)

plt.savefig(
    '3_cs_feature_lollipop.png',
    dpi=300,
    facecolor=fig3.get_facecolor(),
    bbox_inches='tight',
)
plt.show()