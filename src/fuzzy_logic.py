import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_fuzzy_system():
    """
    Defines the Linguistic Variables and Rules for Credit Risk Assessment.
    """
    # 1. Define Variables (Antecedents & Consequent)
    # Universe of discourse: 0 to 1 for inputs, 0 to 100 for output risk
    anomaly_score = ctrl.Antecedent(np.arange(0, 1.05, 0.05), 'anomaly_score')
    ext_source = ctrl.Antecedent(np.arange(0, 1.05, 0.05), 'ext_source')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # 2. Membership Functions (The "Fuzzy" definitions)
    
    # Anomaly Score (From Autoencoder)
    anomaly_score['low'] = fuzz.trimf(anomaly_score.universe, [0, 0, 0.4])
    anomaly_score['medium'] = fuzz.trimf(anomaly_score.universe, [0.2, 0.5, 0.8])
    anomaly_score['high'] = fuzz.trapmf(anomaly_score.universe, [0.6, 0.8, 1, 1])

    # External Source Score (Higher is Better/Safer)
    ext_source['low'] = fuzz.trapmf(ext_source.universe, [0, 0, 0.3, 0.5])
    ext_source['medium'] = fuzz.trimf(ext_source.universe, [0.3, 0.6, 0.8])
    ext_source['high'] = fuzz.trimf(ext_source.universe, [0.6, 1, 1])

    # Risk Level (Output)
    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 80])
    risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

    # 3. Fuzzy Rules (Expert Knowledge Base)
    
    # Rule 1: High Anomaly (AI detected weird behavior) -> High Risk
    rule1 = ctrl.Rule(anomaly_score['high'], risk['high'])
    
    # Rule 2: Low External Score (Bad credit history) -> High Risk
    rule2 = ctrl.Rule(ext_source['low'], risk['high'])
    
    # Rule 3: Low Anomaly AND High Ext Source -> Low Risk (Safe Customer)
    rule3 = ctrl.Rule(anomaly_score['low'] & ext_source['high'], risk['low'])
    
    # Rule 4: Medium Anomaly OR Medium Ext Source -> Medium Risk
    rule4 = ctrl.Rule(anomaly_score['medium'] | ext_source['medium'], risk['medium'])

    # 4. Build System
    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)
    
    return risk_simulation