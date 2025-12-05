import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

# ================== 1. 核心特征工程类 (保持不变) ==================
class PolymerFeature:
    """计算聚合物特征（支持单个/批量处理，返回加权和 & 加权平均值）"""
    def __init__(self):
        self.desc_list = [
            "LogP", "TPSA", "HBA", "HBD", "AromaticRings",
            "MW", "HeavyAtomCount", "RotatableBonds", "HeteroatomCount",
            "RingCount", "AliphaticRings", "SaturatedRings", "AromaticAtoms",
            "FormalCharge", "FractionCSP3",
            "NumAmideBonds", "DoubleBondCount",
            "LabuteASA", "VSA_Estate1", "VSA_Estate2",
            "NumRingsSharingAtoms", "NumBicyclicAtoms"
        ]
        self.monomer_smiles = {
            "EG": "[*]CCC[*]",
            "CL": "O=C([*])CCCCCO[*]",
            "LA": "O=C(C(O[*])C)[*]",
            "LLA": "O=C(C(O[*])C)[*]",
            "DLA": "O=C(C(O[*])C)[*]",
            "TMC": "O=C([*])OCCCO[*]",
            "GA": "O=C(C(O[*])C)[*]",
            "PDO": "O=C([*])COCCO[*]",
            "TOSUO": "O=C([*])CCC1(OCCO1)CCO[*]", 
            "PG": "CC(CC[*])O[*]", 
        }

    def get_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return {}
        desc = {
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "MW": Descriptors.MolWt(mol),
            "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
            "RotatableBonds": Lipinski.NumRotatableBonds(mol),
            "HeteroatomCount": rdMolDescriptors.CalcNumHeteroatoms(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "AliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "SaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
            "FormalCharge": Chem.GetFormalCharge(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),
            "DoubleBondCount": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
            "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
            "NumRingsSharingAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            "NumBicyclicAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        }
        return {k: (v if v is not None else 0) for k, v in desc.items()}

    def parse_polymer(self, polymer_str):
        pattern = r"\((.*?)\)([\d\.]+)"
        matches = re.findall(pattern, polymer_str)
        return {monomer: float(num) for monomer, num in matches}

    def polymer_features(self, polymer_str):
        hydrophilic = {'EG'}
        hydrophobic = {'CL','LA','LLA','DLA','TMC','GA','PDO','TOSUO','PG'}
        composition = self.parse_polymer(polymer_str)
        total_units = sum(composition.values())
        if total_units == 0: return {}

        weighted_A, weighted_B = {}, {}
        for mono, num in composition.items():
            if mono not in self.monomer_smiles: continue
            smiles = self.monomer_smiles[mono]
            desc = self.get_descriptors(smiles)
            for k, v in desc.items():
                if mono in hydrophilic:
                    weighted_A[k] = weighted_A.get(k, 0) + v * num
                elif mono in hydrophobic:
                    weighted_B[k] = weighted_B.get(k, 0) + v * num

        all_features = {}
        all_features.update({f"{k}_A_sum": v for k, v in weighted_A.items()})
        all_features.update({f"{k}_B_sum": v for k, v in weighted_B.items()})
        all_features.update({f"{k}_A_avg": v / total_units for k, v in weighted_A.items()})
        all_features.update({f"{k}_B_avg": v / total_units for k, v in weighted_B.items()})
        return all_features

    def add_polymer_features_to_df(self, df, polymer_col='StruD'):
        polymer_features_list = []
        for poly_str in df[polymer_col]:
            try:
                feats = self.polymer_features(poly_str)
                feats_full = {}
                for desc in self.desc_list:
                    for seg in ['A', 'B']:
                        feats_full[f'Polymer_{desc}_{seg}_sum'] = feats.get(f'{desc}_{seg}_sum', 0)
                        feats_full[f'Polymer_{desc}_{seg}_avg'] = feats.get(f'{desc}_{seg}_avg', 0)
                polymer_features_list.append(feats_full)
            except:
                polymer_features_list.append({})
        poly_feat_df = pd.DataFrame(polymer_features_list)
        return pd.concat([df.reset_index(drop=True), poly_feat_df], axis=1)

# ================== 2. Streamlit 页面逻辑 ==================
st.set_page_config(page_title="AI 水凝胶预测系统", layout="wide")

# 辅助数据：单体重复单元的近似分子量 (用于将 Mn 转换为 聚合度 DP)
# 注意：这里使用常见重复单元的分子量，如有偏差可在此修正
MONOMER_MW = {
    "EG": 44.05,
    "CL": 114.14,
    "LA": 72.06,  # 这里的LA指乳酸单元
    "LLA": 72.06,
    "DLA": 72.06,
    "GA": 58.04,
    "PDO": 102.09,
    "TMC": 102.09,
    "TOSUO": 172.18, # 估算值 C8H12O4
    "PG": 74.08,
    "None": 1.0 
}

@st.cache_resource
def load_models():
    try:
        model = joblib.load("XGB_model.joblib")
        preprocessor = joblib.load("preprocessor.joblib")
        return model, preprocessor
    except FileNotFoundError:
        st.error("未找到模型文件！请确保 'XGB_model.joblib' 和 'preprocessor.joblib' 在当前目录下。")
        return None, None

model, preprocessor = load_models()
pf = PolymerFeature()

st.title("AI 水凝胶相变预测系统")

# --- 侧边栏：输入聚合物固有属性 ---
with st.sidebar:
    st.header("1. 聚合物属性输入")
    st.info("请输入各嵌段信息，系统将自动生成结构式和比例")

    # 1. 拓扑结构
    topology = st.selectbox("拓扑结构 (Topology)", ["BAB", "ABA"], index=0)

    # 2. A 嵌段输入
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        mono_a = st.selectbox("A 单体", ["EG"], index=0, help="亲水段通常为EG")
    with col_a2:
        mn_a_val = st.number_input("A 分子量 (Mn)", value=1000.0, step=100.0)

    # 3. B1 嵌段输入
    col_b1_1, col_b1_2 = st.columns(2)
    with col_b1_1:
        mono_b1 = st.selectbox("B1 单体", ["CL", "LA", "GA", "PDO", "TOSUO", "TMC"], index=0)
    with col_b1_2:
        mn_b1_val = st.number_input("B1 分子量 (Mn)", value=700.0, step=100.0)

    # 4. B2 嵌段输入
    col_b2_1, col_b2_2 = st.columns(2)
    with col_b2_1:
        mono_b2 = st.selectbox("B2 单体", ["None", "CL", "LA", "GA", "PDO", "TOSUO", "TMC"], index=0)
    with col_b2_2:
        mn_b2_val = st.number_input("B2 分子量 (Mn)", value=0.0, step=100.0)

    # 5. GPC 和 PDI
    col_gpc1, col_gpc2 = st.columns(2)
    with col_gpc1:
        gpc = st.number_input("GPC (Mn)", value=2500.0)
    with col_gpc2:
        pdi = st.number_input("PDI", value=1.2)

    # --- 自动计算逻辑 (UPDATED) ---
    
    # 1. 计算各单体聚合度 (DP)
    # round() 四舍五入取整
    dp_a = int(round(mn_a_val / MONOMER_MW.get(mono_a, 100)))
    dp_b1 = int(round(mn_b1_val / MONOMER_MW.get(mono_b1, 100)))
    dp_b2 = int(round(mn_b2_val / MONOMER_MW.get(mono_b2, 100))) if mono_b2 != "None" else 0

    # 2. 生成 StruD 字符串 & 计算总聚合度
    if topology == "ABA":
        # 结构：A - (B1+B2) - A
        # A(a)B1(b1)B2(b2)A(a)
        
        # 字符串构建
        b_part_str = f"({mono_b1}){dp_b1}"
        if mono_b2 != "None" and dp_b2 > 0:
            b_part_str += f"({mono_b2}){dp_b2}"
        a_part_str = f"({mono_a}){dp_a}"
        stru_d = f"{a_part_str}{b_part_str}{a_part_str}"

        # 汇总聚合度 (Total DP)
        total_dp_a = dp_a * 2
        total_dp_b = dp_b1 + dp_b2
        
        # 总分子量 Mn 计算 (基于输入值)
        calc_mn_total = (mn_a_val * 2) + mn_b1_val + mn_b2_val

    else: # BAB
        # 结构：(B1+B2) - A - (B1+B2)
        # B1(b1)B2(b2)A(a)B1(b1)B2(b2)
        
        # 字符串构建
        b_part_str = f"({mono_b1}){dp_b1}"
        if mono_b2 != "None" and dp_b2 > 0:
            b_part_str += f"({mono_b2}){dp_b2}"
        a_part_str = f"({mono_a}){dp_a}"
        stru_d = f"{b_part_str}{a_part_str}{b_part_str}"

        # 汇总聚合度 (Total DP)
        total_dp_a = dp_a
        total_dp_b = (dp_b1 + dp_b2) * 2

        # 总分子量 Mn 计算 (基于输入值)
        calc_mn_total = mn_a_val + (mn_b1_val * 2) + (mn_b2_val * 2)

    # 3. 计算 Ratio (严格按照你的公式)
    # Ratio_A = Total A Segments / Total B Segments
    # Ratio_B = Total B Segments / Total A Segments
    
    if total_dp_b > 0:
        calc_ratio_a = total_dp_a / total_dp_b
    else:
        calc_ratio_a = 0.0 # 避免除以零
        
    if total_dp_a > 0:
        calc_ratio_b = total_dp_b / total_dp_a
    else:
        calc_ratio_b = 0.0

    # 显示计算结果预览
    st.markdown("---")
    st.markdown("**🧪 自动生成的结构参数:**")
    st.code(f"StruD: {stru_d}", language="text")
    # 显示聚合度详情，方便核对
    st.caption(f"Total DP_A: {total_dp_a} | Total DP_B: {total_dp_b}")
    st.caption(f"Ratio_A (A/B): {calc_ratio_a:.3f} | Ratio_B (B/A): {calc_ratio_b:.3f}")
    st.caption(f"Total Mn: {calc_mn_total:.1f}")


# --- 主界面：调节实验条件 ---
st.header("实验条件调节 & 实时预测")

col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    # 使用滑块调节温度和浓度，模拟寻找相变点
    temperature = st.slider("温度 (Temperature, °C)", min_value=0.0, max_value=80.0, value=37.0)
    concentration = st.slider("浓度 (Concentration, wt%)", min_value=1.0, max_value=50.0, value=20.0)

    # 构造输入 DataFrame (使用自动计算的值)
    input_data = {
        'StruD': [stru_d],
        'Topology': [topology],
        'Mn': [calc_mn_total], # 使用计算出的总 Mn
        'GPC': [gpc],
        'PDI': [pdi],
        'Concentration_wt%': [concentration],
        'Temperature': [temperature],
        'Ratio_A': [calc_ratio_a], # 使用计算出的 Ratio_A
        'Ratio_B': [calc_ratio_b]  # 使用计算出的 Ratio_B
    }
    df_input = pd.DataFrame(input_data)

    if st.button("开始预测 (Predict)", type="primary"):
        if model and preprocessor:
            try:
                # 1. 特征工程：生成 RDKit 描述符
                df_features = pf.add_polymer_features_to_df(df_input)
                
                # 2. 确保列顺序与训练时一致
                base_features = ['Topology', 'Mn', 'GPC', 'PDI', 'Concentration_wt%', 'Temperature', 'Ratio_A', 'Ratio_B']
                poly_cols = [c for c in df_features.columns if c.startswith('Polymer_')]
                features_all = base_features + poly_cols
                
                X = df_features[features_all]
                
                # 3. 预处理
                X_processed = preprocessor.transform(X)
                
                # 4. 预测
                prediction = model.predict(X_processed)[0]
                probability = model.predict_proba(X_processed)[0]

                # 5. 显示结果
                with col_main2:
                    st.markdown("### 预测结果")
                    if prediction == 1:
                        st.success("## Hydrogel (水凝胶)")
                        st.metric("置信度 (Confidence)", f"{probability[1]*100:.2f}%")
                    else:
                        st.info("## Solution (溶液)")
                        st.metric("置信度 (Confidence)", f"{probability[0]*100:.2f}%")
                    
                    st.markdown("---")
                    st.write("**输入特征快照:**")
                    st.json(input_data)

            except Exception as e:
                st.error(f"预测出错: {str(e)}\n请检查特征列是否与模型匹配。")
        else:
            st.error("模型未加载，无法预测。")

# 添加一个提示框
st.markdown("---")
st.caption("小贴士: 在左侧修改单体类型和分子量，系统会自动计算聚合度并生成结构式。")
