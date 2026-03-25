import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

###############################################################################
# Spirulina PBR Dynamic Simulation
# Midori-no-Tamaki: Feedback loop visualization
#
# State variables: DIC(t), Biomass(t)
# Derived: pH(t), Ci(t), b_env(t), growth_rate(t)
###############################################################################

# === Physical Constants ===
R_gas = 8.314e-3  # kJ/mol/K

# === Carbonate Equilibrium ===
def carbonate_K1(T_C):
    """First dissociation constant of carbonic acid, temperature dependent"""
    T_K = T_C + 273.15
    # Harned & Davis approximation
    pK1 = 6.352 - 0.0152 * T_C  # simplified
    return 10**(-pK1)

def carbonate_K2(T_C):
    """Second dissociation constant"""
    pK2 = 10.33 - 0.0100 * T_C  # simplified
    return 10**(-pK2)

def carbonate_fractions(pH, T_C):
    """Returns (alpha0, alpha1, alpha2) = fractions of CO2, HCO3-, CO3--"""
    H = 10**(-pH)
    K1 = carbonate_K1(T_C)
    K2 = carbonate_K2(T_C)
    
    denom0 = 1 + K1/H + K1*K2/H**2
    denom1 = H/K1 + 1 + K2/H
    denom2 = H**2/(K1*K2) + H/K2 + 1
    
    return 1/denom0, 1/denom1, 1/denom2

def DIC_to_pH(DIC_mM, alkalinity_mM, T_C):
    """
    Compute pH from DIC and alkalinity using charge balance.
    Alkalinity ≈ [HCO3-] + 2[CO3--] + [OH-] - [H+]
    Iterative solver (Newton's method).
    """
    K1 = carbonate_K1(T_C)
    K2 = carbonate_K2(T_C)
    Kw = 1e-14  # water autoionization
    
    # Convert to mol/L
    DIC = DIC_mM * 1e-3
    Alk = alkalinity_mM * 1e-3
    
    # Newton iteration
    pH_guess = 9.0
    for _ in range(50):
        H = 10**(-pH_guess)
        a0, a1, a2 = carbonate_fractions(pH_guess, T_C)
        
        # Alkalinity = DIC*(a1 + 2*a2) + Kw/H - H
        Alk_calc = DIC * (a1 + 2*a2) + Kw/H - H
        
        # Derivative of Alk w.r.t. pH (numerical)
        dpH = 0.001
        H2 = 10**(-(pH_guess + dpH))
        _, a1p, a2p = carbonate_fractions(pH_guess + dpH, T_C)
        Alk_calc2 = DIC * (a1p + 2*a2p) + Kw/H2 - H2
        
        dAlk_dpH = (Alk_calc2 - Alk_calc) / dpH
        
        if abs(dAlk_dpH) < 1e-20:
            break
            
        pH_guess = pH_guess - (Alk_calc - Alk) / dAlk_dpH
        pH_guess = np.clip(pH_guess, 5, 14)
    
    return pH_guess

# === Photosynthesis Model (EOS-based) ===
def b_thermal_calc(T_C, Ea_r=63.0, T_opt=35.0, Tm=42.0, dT=3.0):
    """D1 repair capacity relative to optimal"""
    T_K = T_C + 273.15
    T_opt_K = T_opt + 273.15
    
    # Arrhenius cold branch
    kr_ratio = np.exp((Ea_r / R_gas) * (1/T_opt_K - 1/T_K))
    
    # Denaturation sigmoid
    denaturing = 1 / (1 + np.exp((T_C - Tm) / dT))
    
    return kr_ratio * denaturing

def b_carbon_calc(pH, DIC_mM, T_C, Km_mM=0.5):
    """Carbon supply limitation (Michaelis-Menten on Ci_eff)"""
    _, alpha1, _ = carbonate_fractions(pH, T_C)
    Ci_eff = DIC_mM * alpha1  # HCO3- as primary C source
    
    # CCM concentrates ~1000x, but we fold that into effective Km
    # Effective Km = 0.5 mM HCO3- (after accounting for CCM)
    return Ci_eff / (Km_mM + Ci_eff)

def photosynthesis_rate(I, T_C, pH, DIC_mM, biomass_gL, params):
    """
    Gross photosynthesis rate [mM C / h]
    """
    # Light per cell (Beer-Lambert attenuation)
    k_ext = params['k_ext']  # extinction coefficient [L/g/cm]
    path_length = params['path_length']  # cm
    I_avg = I * (1 - np.exp(-k_ext * biomass_gL * path_length)) / \
            (k_ext * biomass_gL * path_length + 1e-10)
    
    # Maximum photosynthesis at optimal conditions
    Pmax = params['Pmax_opt']  # mM C / g biomass / h
    
    # Light response (PCC channel - tanh saturation)
    Ik = params['Ik']  # light saturation parameter
    PCC = np.tanh(I_avg / Ik)
    
    # Temperature limitation
    b_thermal = b_thermal_calc(T_C)
    
    # Carbon limitation
    b_carbon = b_carbon_calc(pH, DIC_mM, T_C)
    
    # Effective rate = Pmax * PCC * min(b_thermal, b_carbon) * biomass
    b_eff = min(b_thermal, b_carbon)
    
    Pgross = Pmax * PCC * b_eff * biomass_gL
    return Pgross, b_thermal, b_carbon, I_avg

def respiration_rate(T_C, biomass_gL, params):
    """Dark respiration rate [mM C / h]"""
    R_ref = params['R_ref']  # at T_ref
    T_ref = params['T_ref']
    Ea_resp = params['Ea_resp']
    
    T_K = T_C + 273.15
    T_ref_K = T_ref + 273.15
    
    R = R_ref * np.exp((Ea_resp / R_gas) * (1/T_ref_K - 1/T_K))
    return R * biomass_gL

def CO2_supply(pH, DIC_mM, T_C, params):
    """CO2 gas transfer rate [mM C / h]"""
    if not params['CO2_on']:
        return 0.0
    
    kLa = params['kLa']  # h^-1
    CO2_sat = params['CO2_sat']  # mM (saturation at given pCO2)
    
    # Current dissolved CO2
    a0, _, _ = carbonate_fractions(pH, T_C)
    CO2_current = DIC_mM * a0
    
    # Gas transfer (CO2 dissolution adds to DIC)
    return kLa * (CO2_sat - CO2_current)

# === Growth Model ===
def growth_rate(Pgross, R, biomass_gL, params):
    """Net specific growth rate [h^-1]"""
    eta = params['eta_growth']  # allocation efficiency to growth
    C_per_biomass = params['C_per_biomass']  # mM C per g biomass
    
    net_C = Pgross * eta - R
    mu = net_C / (C_per_biomass * biomass_gL + 1e-10)
    return max(mu, -0.01)  # prevent unrealistic negative growth

# === Simulation ===
def simulate_PBR(params, T_hours=400, dt=0.1):
    """Run PBR dynamics simulation"""
    
    n_steps = int(T_hours / dt)
    
    # State arrays
    t = np.zeros(n_steps)
    DIC = np.zeros(n_steps)
    biomass = np.zeros(n_steps)
    pH_arr = np.zeros(n_steps)
    Pgross_arr = np.zeros(n_steps)
    R_arr = np.zeros(n_steps)
    b_thermal_arr = np.zeros(n_steps)
    b_carbon_arr = np.zeros(n_steps)
    I_avg_arr = np.zeros(n_steps)
    mu_arr = np.zeros(n_steps)
    
    # Initial conditions
    DIC[0] = params['DIC_init']
    biomass[0] = params['biomass_init']
    alkalinity = params['alkalinity']  # constant (conservative)
    T_C = params['temperature']
    I = params['irradiance']
    
    # Initial pH
    pH_arr[0] = DIC_to_pH(DIC[0], alkalinity, T_C)
    
    for i in range(1, n_steps):
        t[i] = i * dt
        
        # Light cycle (16:8 or continuous)
        if params.get('light_cycle', False):
            hour_of_day = (t[i] % 24)
            I_now = I if hour_of_day < 16 else 0
        else:
            I_now = I
        
        # Current state
        DIC_now = DIC[i-1]
        bio_now = biomass[i-1]
        pH_now = pH_arr[i-1]
        
        # Rates
        Pg, bt, bc, Ia = photosynthesis_rate(I_now, T_C, pH_now, DIC_now, bio_now, params)
        R = respiration_rate(T_C, bio_now, params)
        CO2_in = CO2_supply(pH_now, DIC_now, T_C, params)
        
        # DIC dynamics: consumed by photosynthesis, produced by respiration, supplied by gas
        dDIC = -Pg + R + CO2_in
        
        # Biomass dynamics
        mu = growth_rate(Pg, R, bio_now, params)
        dBio = mu * bio_now
        
        # Update state
        DIC[i] = max(DIC_now + dDIC * dt, 0.1)  # prevent negative DIC
        biomass[i] = max(bio_now + dBio * dt, 0.01)
        
        # Compute new pH from DIC and alkalinity
        pH_arr[i] = DIC_to_pH(DIC[i], alkalinity, T_C)
        
        # Store
        Pgross_arr[i] = Pg
        R_arr[i] = R
        b_thermal_arr[i] = bt
        b_carbon_arr[i] = bc
        I_avg_arr[i] = Ia
        mu_arr[i] = mu
    
    return {
        't': t, 'DIC': DIC, 'biomass': biomass, 'pH': pH_arr,
        'Pgross': Pgross_arr, 'R': R_arr, 
        'b_thermal': b_thermal_arr, 'b_carbon': b_carbon_arr,
        'I_avg': I_avg_arr, 'mu': mu_arr
    }

# === Parameter Sets ===
base_params = {
    # Culture conditions
    'DIC_init': 200.0,       # mM (Zarrouk medium: 16.8 g/L NaHCO3)
    'alkalinity': 200.0,     # mM (from NaHCO3)
    'biomass_init': 0.1,     # g/L
    'temperature': 35.0,     # °C
    'irradiance': 250.0,     # µmol/m²/s
    
    # Photosynthesis parameters
    'Pmax_opt': 2.0,         # mM C / g biomass / h (at optimal T)
    'Ik': 150.0,             # µmol/m²/s (light saturation)
    'k_ext': 0.02,           # L/g/cm (extinction coefficient)
    'path_length': 4.0,      # cm (tube diameter)
    
    # Respiration
    'R_ref': 0.3,            # mM C / g biomass / h at T_ref
    'T_ref': 35.0,           # °C
    'Ea_resp': 48.8,         # kJ/mol (Torzillo 1994)
    
    # CO2 supply
    'CO2_on': False,         # No CO2 control
    'kLa': 5.0,              # h^-1 (gas transfer coefficient)
    'CO2_sat': 0.15,         # mM (5% CO2 in air, at 35°C)
    
    # Growth
    'eta_growth': 0.6,       # allocation to growth
    'C_per_biomass': 40.0,   # mM C per g biomass (~480 mg C/g)
    
    # Light cycle
    'light_cycle': False,    # Continuous light
}

# === Run Scenarios ===
print("Running simulations...")

# Scenario 1: No pH control (Kobayashi 1996 replication)
params_1 = {**base_params, 'CO2_on': False, 'temperature': 25.0}
result_1 = simulate_PBR(params_1, T_hours=300)

# Scenario 2: pH control via CO2
params_2 = {**base_params, 'CO2_on': True, 'temperature': 25.0}
result_2 = simulate_PBR(params_2, T_hours=300)

# Scenario 3: pH control + optimal temperature (35°C)
params_3 = {**base_params, 'CO2_on': True, 'temperature': 35.0}
result_3 = simulate_PBR(params_3, T_hours=300)

# Scenario 4: Suboptimal temperature, no CO2
params_4 = {**base_params, 'CO2_on': False, 'temperature': 25.0, 'irradiance': 500.0}
result_4 = simulate_PBR(params_4, T_hours=300)

print("Simulations complete!")

# === Plotting ===
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Spirulina PBR Dynamic Simulation: Feedback Loop Behavior\n'
             'b_env = min(b_thermal, b_carbon); β_eff = β_ref/b_env', 
             fontsize=14, fontweight='bold', y=0.99)

colors = {'s1': '#e74c3c', 's2': '#3498db', 's3': '#2ecc71', 's4': '#e67e22'}
labels = {
    's1': '25°C, no CO₂ (Kobayashi replication)',
    's2': '25°C, CO₂ controlled',
    's3': '35°C, CO₂ controlled (optimal)',
    's4': '25°C, no CO₂, high light (500 µE)',
}

# Panel A: Biomass
ax = axes[0, 0]
for key, res, c in [('s1', result_1, colors['s1']), ('s2', result_2, colors['s2']), 
                      ('s3', result_3, colors['s3']), ('s4', result_4, colors['s4'])]:
    ax.plot(res['t'], res['biomass'], color=c, linewidth=2, label=labels[key])
ax.set_ylabel('Biomass (g/L)', fontsize=12)
ax.set_title('A - Biomass Growth', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0, 300)
# Kobayashi reference lines
ax.axhline(y=1.2, color=colors['s1'], linestyle=':', alpha=0.5)
ax.axhline(y=2.2, color=colors['s2'], linestyle=':', alpha=0.5)
ax.text(250, 1.3, '1.2 g/L\n(Kobayashi)', fontsize=7, color=colors['s1'])

# Panel B: pH
ax = axes[0, 1]
for key, res, c in [('s1', result_1, colors['s1']), ('s2', result_2, colors['s2']), 
                      ('s3', result_3, colors['s3']), ('s4', result_4, colors['s4'])]:
    ax.plot(res['t'], res['pH'], color=c, linewidth=2)
ax.axhspan(8.5, 10.5, alpha=0.1, color='green', label='Optimal pH range')
ax.axhline(y=12.0, color='red', linestyle='--', alpha=0.7, label='pH 12: death zone')
ax.set_ylabel('pH', fontsize=12)
ax.set_title('B - pH Dynamics (the runaway!)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0, 300)
ax.set_ylim(8, 13)

# Panel C: DIC
ax = axes[1, 0]
for key, res, c in [('s1', result_1, colors['s1']), ('s2', result_2, colors['s2']), 
                      ('s3', result_3, colors['s3']), ('s4', result_4, colors['s4'])]:
    ax.plot(res['t'], res['DIC'], color=c, linewidth=2)
ax.set_ylabel('DIC (mM)', fontsize=12)
ax.set_title('C - Dissolved Inorganic Carbon', fontsize=12, fontweight='bold')
ax.set_xlim(0, 300)

# Panel D: b_carbon vs b_thermal
ax = axes[1, 1]
for key, res, c in [('s1', result_1, colors['s1']), ('s2', result_2, colors['s2']), 
                      ('s3', result_3, colors['s3']), ('s4', result_4, colors['s4'])]:
    ax.plot(res['t'], res['b_carbon'], color=c, linewidth=2, linestyle='-')
    ax.plot(res['t'], res['b_thermal'], color=c, linewidth=1.5, linestyle='--', alpha=0.6)
ax.set_ylabel('β value', fontsize=12)
ax.set_title('D - b_carbon (solid) vs b_thermal (dashed)', fontsize=12, fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.text(5, 0.52, 'Bottleneck threshold', fontsize=8, color='gray')
ax.set_xlim(0, 300)
ax.set_ylim(0, 1.1)

# Panel E: Bottleneck diagnosis
ax = axes[2, 0]
for key, res, c, label in [('s1', result_1, colors['s1'], labels['s1']),
                             ('s3', result_3, colors['s3'], labels['s3'])]:
    # Which is the bottleneck?
    bottleneck = np.where(res['b_carbon'] < res['b_thermal'], 
                          res['b_carbon'], res['b_thermal'])
    is_carbon = res['b_carbon'] < res['b_thermal']
    
    ax.fill_between(res['t'], 0, 1, where=is_carbon, alpha=0.2, color='blue', 
                    label='b_carbon dominant' if key=='s1' else '')
    ax.fill_between(res['t'], 0, 1, where=~is_carbon, alpha=0.2, color='orange',
                    label='b_thermal dominant' if key=='s1' else '')
    ax.plot(res['t'], bottleneck, color=c, linewidth=2, label=label[:30])

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('b_env (bottleneck capacity)', fontsize=12)
ax.set_title('E - Bottleneck Switching', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='lower left')
ax.set_xlim(0, 300)
ax.set_ylim(0, 1.1)

# Panel F: Growth rate
ax = axes[2, 1]
for key, res, c in [('s1', result_1, colors['s1']), ('s2', result_2, colors['s2']), 
                      ('s3', result_3, colors['s3']), ('s4', result_4, colors['s4'])]:
    # Smooth growth rate for display
    window = 50
    if len(res['mu']) > window:
        mu_smooth = np.convolve(res['mu'], np.ones(window)/window, mode='same')
    else:
        mu_smooth = res['mu']
    ax.plot(res['t'], mu_smooth * 24, color=c, linewidth=2)  # convert to day^-1

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Growth rate (day⁻¹)', fontsize=12)
ax.set_title('F - Specific Growth Rate', fontsize=12, fontweight='bold')
ax.set_xlim(0, 300)
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/mnt/user-data/outputs/Fig_PBR_dynamics_simulation.png', dpi=150, bbox_inches='tight')
print("Figure saved!")

# Print key results
print("\n" + "="*60)
print("SIMULATION RESULTS SUMMARY")
print("="*60)
for name, res, params in [
    ("S1: 25°C, no CO2", result_1, params_1),
    ("S2: 25°C, +CO2",   result_2, params_2),
    ("S3: 35°C, +CO2",   result_3, params_3),
    ("S4: 25°C, no CO2, 500µE", result_4, params_4),
]:
    max_bio = np.max(res['biomass'])
    final_pH = res['pH'][-1]
    min_bcarbon = np.min(res['b_carbon'][100:])  # skip initial
    btherm = res['b_thermal'][100] if len(res['b_thermal']) > 100 else res['b_thermal'][-1]
    print(f"\n  {name}:")
    print(f"    Max biomass: {max_bio:.2f} g/L")
    print(f"    Final pH: {final_pH:.1f}")
    print(f"    Min b_carbon: {min_bcarbon:.3f}")
    print(f"    b_thermal: {btherm:.3f}")
    bottleneck = "CARBON" if min_bcarbon < btherm else "THERMAL"
    print(f"    Bottleneck: {bottleneck}")

