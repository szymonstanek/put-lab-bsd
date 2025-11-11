import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
FREQUENCY_MHZ = 541.9
FREQUENCY_HZ = FREQUENCY_MHZ * 1e6
X0 = 13
Y0 = 27
RX_FLOOR = 0
FLOOR_HEIGHT = 4  # H in Eq. 3
NOISE_LEVEL_1_25 = -100
NOISE_LEVEL_26_95 = -107
THRESHOLD_1_25 = -90
THRESHOLD_26_95 = -97
C = 3e8

print("="*70)
print("KALIBRACJA EMPIRYCZNEGO MODELU PROPAGACYJNEGO")
print("="*70)
print(f"\nCzęstotliwość: {FREQUENCY_MHZ} MHz")
print(f"Pozycja odbiornika: ({X0}, {Y0}) m, Piętro {RX_FLOOR}")
print(f"Wysokość piętra: {FLOOR_HEIGHT} m")

df = pd.read_excel('pomiary.xlsx')
print(f"\nWczytano pomiarów: {len(df)}")

df_excluded = df[df['P_RX_dBm'].isna()].copy()
if len(df_excluded) > 0:
    df_excluded['Point_Number'] = df_excluded.index + 1
    excluded_table = df_excluded[['Point_Number', 'x', 'y', 'floor']].copy()
    excluded_table.columns = ['Point #', 'x [m]', 'y [m]', 'Floor']
    excluded_table.to_excel('excluded_measurements.xlsx', index=False)
    print(f"Wykluczono pomiarów: {len(df_excluded)}")

df_clean = df.dropna(subset=['P_RX_dBm']).copy()
N = len(df_clean)
print(f"Prawidłowych pomiarów: {N}")

x_i = df_clean['x'].values
y_i = df_clean['y'].values
n_i = df_clean['floor'].values
y_i_measured = df_clean['P_RX_dBm'].values

print(f"Piętro 0: {np.sum(n_i == 0)}, Piętro 1: {np.sum(n_i == 1)}, Piętro 2: {np.sum(n_i == 2)} pomiarów")

print(f"\n{'='*70}")
print("OBLICZANIE ODLEGŁOŚCI")
print("="*70)

# Eq. 3
d_i = np.sqrt((x_i - X0)**2 + (y_i - Y0)**2 + (n_i * FLOOR_HEIGHT)**2)

print(f"Odległość min: {np.min(d_i):.2f} m, max: {np.max(d_i):.2f} m")

print(f"\n{'='*70}")
print("ESTYMACJA PARAMETRÓW (Metoda najmniejszych kwadratów)")
print("="*70)

# Eq. 4: y_i = A - γ * log10(d_i) - n_i * ΔL
# Eq. 5: Y = X * wsp
X_matrix = np.column_stack([
    np.ones(N),
    -np.log10(d_i),
    -n_i
])

Y_vector = y_i_measured

# Eq. 6
wsp, residuals, rank, s = np.linalg.lstsq(X_matrix, Y_vector, rcond=None)

A_estimated = wsp[0]
gamma_estimated = wsp[1]
delta_L_estimated = wsp[2]

print(f"\nParametry estymowane:")
print(f"  A = {A_estimated:.2f} dBm")
print(f"  γ = {gamma_estimated:.2f}")
print(f"  ΔL = {delta_L_estimated:.2f} dB")

print(f"\n{'='*70}")
print("PREDYKCJA I ANALIZA BŁĘDU")
print("="*70)

y_i_predicted = X_matrix @ wsp
e_i = y_i_measured - y_i_predicted
mean_error = np.mean(e_i)
std_error = np.std(e_i, ddof=1)

print(f"\nŚredni błąd: {mean_error:.2f} dB")
print(f"Odchylenie standardowe: {std_error:.2f} dB")

print(f"\n{'='*70}")
print("PORÓWNANIE Z REKOMENDACJĄ ITU-R P.1238")
print("="*70)

print(f"\nCzęstotliwość pomiaru: {FREQUENCY_MHZ} MHz ({FREQUENCY_MHZ/1000:.3f} GHz)")
print(f"Referencja: ITU-R P.1238-10 Tabele 2, 3, i 4")

print(f"\n{'-'*70}")
print("TABELA 2: Współczynnik tłumienia odległościowego (N = γ)")
print(f"{'-'*70}")
print(f"Estymowane γ: {gamma_estimated:.2f}")
print(f"\nWartości ITU-R P.1238 Tabela 2:")
print(f"\n  0.9 GHz: Office N=33, Commercial N=20")
print(f"  1.25 GHz: Office N=32, Commercial N=22")
print(f"  1.9 GHz: Residential N=28, Office N=30, Commercial N=22")

print(f"\n{'-'*70}")
print("TABELA 3: Tłumienie międzypiętrowe (ΔL)")
print(f"{'-'*70}")
print(f"Estymowane ΔL: {delta_L_estimated:.2f} dB")
print(f"\nWartości ITU-R P.1238 Tabela 3:")
print(f"  1.8-2.0 GHz: Residential 4 dB, Office 15+4(n-1) dB, Commercial 6+3(n-1) dB")
print(f"  → Najbliższe: środowisko RESIDENTIAL (4-8 dB)")

print(f"\n{'-'*70}")
print("TABELA 4: Odchylenie standardowe cieniowania (σ)")
print(f"{'-'*70}")
print(f"Estymowane σ: {std_error:.2f} dB")
print(f"\nWartości ITU-R P.1238 Tabela 4:")
print(f"  1.8-2.0 GHz: Residential σ=8 dB, Office σ=10 dB, Commercial σ=10 dB")
print(f"  → Najbliższe: środowisko OFFICE/COMMERCIAL (8-12 dB)")

print(f"\n{'-'*70}")
print("KLASYFIKACJA ŚRODOWISKA")
print(f"{'-'*70}")
print(f"\nNa podstawie ITU-R P.1238:")
print(f"  ΔL = {delta_L_estimated:.2f} dB → RESIDENTIAL")
print(f"  σ = {std_error:.2f} dB → OFFICE/COMMERCIAL")
print(f"  γ = {gamma_estimated:.2f} → ANOMALNE (wymaga weryfikacji)")
print(f"\nNajprawdopodobniej: budynek RESIDENTIAL lub COMMERCIAL")

print(f"\n{'='*70}")
print("GENEROWANIE WYKRESÓW")
print("="*70)

floors = [0, 1, 2]
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

print("  Wykres 1: Moc vs Odległość...")
floor_names = ['Parter (Piętro 0)', 'Piętro 1', 'Piętro 2']

for floor, color, marker, floor_name in zip(floors, colors, markers, floor_names):
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111)

    mask = n_i == floor

    # Measured values for this floor
    ax1.scatter(d_i[mask], y_i_measured[mask],
               c=color, marker=marker, s=100, alpha=0.7,
               label=f'Pomiary zmierzone', edgecolors='black', linewidth=1)

    # Predicted model line for this floor
    d_range = np.linspace(np.min(d_i[mask]), np.max(d_i[mask]), 200)
    y_pred_line = A_estimated - gamma_estimated * np.log10(d_range) - floor * delta_L_estimated
    ax1.plot(d_range, y_pred_line, color=color, linewidth=3,
            linestyle='--', alpha=0.9, label=f'Model predykcyjny')

    ax1.set_xscale('log')
    ax1.set_xlabel('Odległość d_i [m]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Moc odebrana P_RX [dBm]', fontsize=14, fontweight='bold')
    ax1.set_title(f'Moc odebrana vs Odległość - {floor_name}\nPomiary zmierzone (y_i) i przewidywane (ỹ_i)',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4, which='both', linestyle='--')
    ax1.legend(loc='best', fontsize=12, framealpha=0.95)
    ax1.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.savefig(f'plot1_power_vs_distance_floor_{floor}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
fig1_combined = plt.figure(figsize=(12, 8))
ax1_combined = fig1_combined.add_subplot(111)

for floor, color, marker, floor_name in zip(floors, colors, markers, floor_names):
    mask = n_i == floor
    ax1_combined.scatter(d_i[mask], y_i_measured[mask],
               c=color, marker=marker, s=80, alpha=0.7,
               label=f'Pomiary - {floor_name}', edgecolors='black', linewidth=0.5)

for floor, color, floor_name in zip(floors, colors, floor_names):
    d_range = np.linspace(np.min(d_i), np.max(d_i), 200)
    y_pred_line = A_estimated - gamma_estimated * np.log10(d_range) - floor * delta_L_estimated
    ax1_combined.plot(d_range, y_pred_line, color=color, linewidth=2.5,
            linestyle='--', alpha=0.9, label=f'Model - {floor_name}')

ax1_combined.set_xscale('log')
ax1_combined.set_xlabel('Odległość d_i [m]', fontsize=14, fontweight='bold')
ax1_combined.set_ylabel('Moc odebrana P_RX [dBm]', fontsize=14, fontweight='bold')
ax1_combined.set_title('Moc odebrana vs Odległość - Wszystkie Piętra\nPomiary zmierzone (y_i) i przewidywane (ỹ_i)',
              fontsize=16, fontweight='bold', pad=20)
ax1_combined.grid(True, alpha=0.4, which='both', linestyle='--')
ax1_combined.legend(loc='best', fontsize=10, ncol=2, framealpha=0.95)
ax1_combined.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig('plot1_power_vs_distance_combined.png', dpi=300, bbox_inches='tight')
plt.close(fig1_combined)

print("  Wykres 2: Mapa pomiarów...")
for floor, color, marker, floor_name in zip(floors, colors, markers, floor_names):
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111)

    mask = n_i == floor

    scatter = ax2.scatter(y_i[mask], x_i[mask],
                         c=y_i_measured[mask], marker=marker,
                         s=200, alpha=0.8, cmap='RdYlGn',
                         vmin=-100, vmax=-50,
                         edgecolors='black', linewidth=1.5,
                         label=f'{floor_name}')

    if floor == 0:
        ax2.scatter(Y0, X0, c='black', marker='*', s=1000,
                   edgecolors='yellow', linewidth=3, label='Odbiornik (RX)', zorder=100)

    ax2.set_xlabel('Współrzędna Y [m]', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Współrzędna X [m]', fontsize=14, fontweight='bold')
    ax2.set_title(f'Mapa Pomiarów - {floor_name}\nRozmieszczenie punktów pomiarowych (x_i, y_i)',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax2.set_aspect('equal')
    ax2.tick_params(axis='both', which='major', labelsize=11)

    cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
    cbar.set_label('Moc odebrana P_RX [dBm]', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(f'plot2_measurement_map_floor_{floor}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
fig2_combined = plt.figure(figsize=(12, 10))
ax2_combined = fig2_combined.add_subplot(111)

for floor, color, marker, floor_name in zip(floors, colors, markers, floor_names):
    mask = n_i == floor
    scatter = ax2_combined.scatter(y_i[mask], x_i[mask],
                         c=y_i_measured[mask], marker=marker,
                         s=150, alpha=0.8, cmap='RdYlGn',
                         vmin=-100, vmax=-50,
                         edgecolors='black', linewidth=1,
                         label=floor_name)

ax2_combined.scatter(Y0, X0, c='black', marker='*', s=800,
           edgecolors='yellow', linewidth=3, label='Odbiornik (RX)', zorder=100)

ax2_combined.set_xlabel('Współrzędna Y [m]', fontsize=14, fontweight='bold')
ax2_combined.set_ylabel('Współrzędna X [m]', fontsize=14, fontweight='bold')
ax2_combined.set_title('Mapa Pomiarów - Wszystkie Piętra\nRozmieszczenie punktów pomiarowych (x_i, y_i)',
              fontsize=16, fontweight='bold', pad=20)
ax2_combined.grid(True, alpha=0.4, linestyle='--')
ax2_combined.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax2_combined.set_aspect('equal')
ax2_combined.tick_params(axis='both', which='major', labelsize=11)

cbar = plt.colorbar(scatter, ax=ax2_combined, pad=0.02)
cbar.set_label('Moc odebrana P_RX [dBm]', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('plot2_measurement_map_combined.png', dpi=300, bbox_inches='tight')
plt.close(fig2_combined)

print("  Wykres 3: Histogram błędów...")
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111)

n_bins = 25
counts, bins, patches = ax3.hist(e_i, bins=n_bins, density=True,
                                 alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)

mu, sigma = mean_error, std_error
x_fit = np.linspace(np.min(e_i) - 5, np.max(e_i) + 5, 200)
y_fit = stats.norm.pdf(x_fit, mu, sigma)
ax3.plot(x_fit, y_fit, 'r-', linewidth=3, label=f'Rozkład normalny')

ax3.set_xlabel('Błąd predykcji e_i = y_i - ỹ_i [dB]', fontsize=14, fontweight='bold')
ax3.set_ylabel('Gęstość prawdopodobieństwa', fontsize=14, fontweight='bold')
ax3.set_title('Histogram błędów',
              fontsize=16, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.4, linestyle='--')
ax3.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax3.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig('plot3_error_histogram.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

print("  Wykres 4: Analiza log-normalności...")
fig4 = plt.figure(figsize=(12, 10))

ax4a = plt.subplot(2, 2, 1)
stats.probplot(e_i, dist="norm", plot=ax4a)
ax4a.set_title('Wykres Q-Q', fontsize=12, fontweight='bold')
ax4a.set_xlabel('Kwantyle teoretyczne', fontsize=11)
ax4a.set_ylabel('Kwantyle próbki', fontsize=11)
ax4a.grid(True, alpha=0.3)

ax4b = plt.subplot(2, 2, 2)
n_bins = 25
ax4b.hist(e_i, bins=n_bins, density=True, alpha=0.7, color='lightblue', edgecolor='black')
x_fit = np.linspace(np.min(e_i) - 5, np.max(e_i) + 5, 200)
y_fit = stats.norm.pdf(x_fit, mean_error, std_error)
ax4b.plot(x_fit, y_fit, 'r-', linewidth=2.5, label='Rozkład normalny')
ax4b.set_xlabel('Błąd e_i [dB]', fontsize=11)
ax4b.set_ylabel('Gęstość prawdopodobieństwa', fontsize=11)
ax4b.set_title('Histogram z dopasowaniem', fontsize=12, fontweight='bold')
ax4b.legend(fontsize=10)
ax4b.grid(True, alpha=0.3)

ax4c = plt.subplot(2, 2, 3)
sorted_errors = np.sort(e_i)
empirical_cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
ax4c.plot(sorted_errors, empirical_cdf, 'b-', linewidth=2, label='CDF empiryczna')
theoretical_cdf = stats.norm.cdf(sorted_errors, mean_error, std_error)
ax4c.plot(sorted_errors, theoretical_cdf, 'r--', linewidth=2, label='CDF teoretyczna')
ax4c.set_xlabel('Błąd e_i [dB]', fontsize=11)
ax4c.set_ylabel('Prawdopodobieństwo skumulowane', fontsize=11)
ax4c.set_title('Porównanie CDF', fontsize=12, fontweight='bold')
ax4c.legend(fontsize=10)
ax4c.grid(True, alpha=0.3)

# Test Shapiro-Wilk
ax4d = plt.subplot(2, 2, 4)
ax4d.axis('off')

plt.suptitle('Analiza log-normalności cieniowania',
            fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('plot4_lognormal_shadowing_test.png', dpi=300, bbox_inches='tight')
plt.close(fig4)

print(f"\n{'='*70}")
print("ZAPIS WYNIKÓW")
print("="*70)

results_df = pd.DataFrame({
    'Point': np.arange(1, N+1),
    'x [m]': x_i,
    'y [m]': y_i,
    'Floor': n_i,
    'Distance [m]': d_i,
    'P_measured [dBm]': y_i_measured,
    'P_predicted [dBm]': y_i_predicted,
    'Error [dB]': e_i
})

results_df.to_excel('results_table.xlsx', index=False)
print("Zapisano: results_table.xlsx")

# Test Shapiro-Wilk
print(f"\n{'='*70}")
print("TEST SHAPIRO-WILKA")
print("="*70)

# Wykonanie testu Shapiro-Wilka dla błędów e_i
shapiro_stat, shapiro_p = stats.shapiro(e_i)
print(f"\nStatystyka: {shapiro_stat:.4f}")
print(f"p-value: {shapiro_p:.4f}")

if shapiro_p > 0.05:
    print(f"\nWYNIK: TAK, cieniowanie ma charakter logarytmiczno-normalny")
    print(f"(błędy mają rozkład normalny, p > 0.05)")
else:
    print(f"\nWYNIK: Cieniowanie może odbiegać od rozkładu log-normalnego")
    print(f"(p ≤ 0.05)")

print(f"\n{'='*70}")
print("ANALIZA ZAKOŃCZONA")
print("="*70)
print("\nWygenerowane pliki:")
print("  Wykresy: plot1_*.png, plot2_*.png, plot3_*.png, plot4_*.png")
print("  Dane: results_table.xlsx")
