import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

# ================== 1. 核心特征工程类 (来自你的 ing.py) ==================
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
        # 补充缺失的单体，根据需要可继续添加
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

# 加载模型和处理器
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
    st.info("在这里输入材料的化学结构信息")
    
    stru_d = st.text_input("结构式 (StruD)", value="(CL)700(EG)1000(CL)700", help="格式示例: (CL)700(EG)1000(CL)700")
    topology = st.selectbox("拓扑结构 (Topology)", ["BAB", "ABA", "Linear", "Star"], index=0) # 根据你的实际类别调整
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        mn = st.number_input("Mn (分子量)", value=2500.0)
        pdi = st.number_input("PDI", value=1.2)
    with col_s2:
        gpc = st.number_input("GPC", value=3500.0)
        ratio_a = st.number_input("Ratio_A (亲水比例)", value=0.21, min_value=0.0, max_value=1.0)
        
    ratio_b = st.number_input("Ratio_B (疏水比例)", value=0.79, min_value=0.0, max_value=1.0)

# --- 主界面：调节实验条件 ---
st.header("2. 实验条件调节 & 实时预测")

col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    # 使用滑块调节温度和浓度，模拟寻找相变点
    temperature = st.slider("温度 (Temperature, °C)", min_value=0.0, max_value=80.0, value=37.0)
    concentration = st.slider("浓度 (Concentration, wt%)", min_value=1.0, max_value=50.0, value=20.0)

    # 构造输入 DataFrame
    input_data = {
        'StruD': [stru_d],
        'Topology': [topology],
        'Mn': [mn],
        'GPC': [gpc],
        'PDI': [pdi],
        'Concentration_wt%': [concentration],
        'Temperature': [temperature],
        'Ratio_A': [ratio_a],
        'Ratio_B': [ratio_b]
    }
    df_input = pd.DataFrame(input_data)

    if st.button("开始预测 (Predict)", type="primary"):
        if model and preprocessor:
            try:
                # 1. 特征工程：生成 RDKit 描述符
                df_features = pf.add_polymer_features_to_df(df_input)
                
                # 2. 确保列顺序与训练时一致 (Base features + Polymer features)
                # 这里我们按照 ing.py 里的逻辑重新组装 features 列表
                base_features = ['Topology', 'Mn', 'GPC', 'PDI', 'Concentration_wt%', 'Temperature', 'Ratio_A', 'Ratio_B']
                poly_cols = [c for c in df_features.columns if c.startswith('Polymer_')]
                features_all = base_features + poly_cols
                
                X = df_features[features_all]
                
                # 3. 预处理 (标准化/OneHot)
                X_processed = preprocessor.transform(X)
                
                # 补全可能丢失的列名以便查看（可选）
                # feature_names = preprocessor.get_feature_names_out()
                # X_processed = pd.DataFrame(X_processed, columns=feature_names)

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
                st.error(f"预测出错: {str(e)}\n请检查输入的结构式格式是否正确，或者特征列是否与模型匹配。")
        else:
            st.error("模型未加载，无法预测。")

# 添加一个提示框
st.markdown("---")
st.caption("小贴士: 修改左侧侧边栏的结构参数，或拖动中间的滑块，点击预测按钮即可查看不同条件下的相态变化。")