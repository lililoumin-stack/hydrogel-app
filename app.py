import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

# ================== 全局配置 ==================
st.set_page_config(page_title="AI 水凝胶设计平台", layout="wide", page_icon="������")

# ================== 1. 核心特征工程类 (保持原有逻辑不变) ==================
class PolymerFeature:
    """计算聚合物特征（支持单个/批量处理，返回加权和 & 加权平均值）"""

    def __init__(self):
        self.carbonyl_pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
        self.desc_list = [
            "LogP", "TPSA", "HBA", "HBD", "AromaticRings",
            "MW", "HeavyAtomCount", "RotatableBonds", "HeteroatomCount",
            "RingCount", "AliphaticRings", "SaturatedRings", "AromaticAtoms",
            "FormalCharge", "FractionCSP3",
            "NumAmideBonds", "StereoCenterCount","CarbonylCount",
            "LabuteASA", "VSA_Estate1", "VSA_Estate2",
            "NumRingsSharingAtoms", "NumBicyclicAtoms",
        ]
        self.monomer_smiles = {
            "EG": "[*]CCO[*]",
            "CL": "O=C([*])CCCCCO[*]",
            "LA": "O=C(C(O[*])C)[*]",
            "LLA": "O=C([C@@H](C)O[*])[*]",
            "DLA": "O=C([C@H](C)O[*])[*]",
            "TMC": "O=C([*])OCCCO[*]",
            "GA": "O=C([*])CO[*]",
            "PDO": "O=C([*])COCCO[*]",
            "TOSUO": "O=C([*])CCC1(OCCO1)CCO[*]", 
            "PG": "C(CC[*])O[*]", 
        }
        
    def get_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {k: 0 for k in self.desc_list}

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
            "AromaticAtoms": sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
            "FormalCharge": Chem.GetFormalCharge(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),
            "StereoCenterCount": rdMolDescriptors.CalcNumAtomStereoCenters(mol), 
            "CarbonylCount": len(mol.GetSubstructMatches(self.carbonyl_pattern)),
            "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
            "NumRingsSharingAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            "NumBicyclicAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
            "VSA_Estate1": 0,
            "VSA_Estate2": 0,
        }
        return {k: (v if v is not None else 0) for k, v in desc.items()}

    def parse_polymer(self, polymer_str):
        pattern = r"\((.*?)\)([\d\.]+)"
        matches = re.findall(pattern, polymer_str)
        composition = {}
        for monomer, num_str in matches:
            num = float(num_str)
            if monomer in composition:
                composition[monomer] += num
            else:
                composition[monomer] = num
        return composition

    def polymer_features(self, polymer_str):
        hydrophilic = {'EG'}
        hydrophobic = {'CL','LA','LLA','DLA','TMC','GA','PDO','TOSUO','PG'}
        composition = self.parse_polymer(polymer_str)
        total_units = sum(composition.values())
        if total_units == 0: raise ValueError("总聚合度为0")

        dp_A = sum(num for mono, num in composition.items() if mono in hydrophilic)
        dp_B = sum(num for mono, num in composition.items() if mono in hydrophobic)

        weighted_A = {k: 0 for k in self.desc_list}
        weighted_B = {k: 0 for k in self.desc_list}

        for mono, num in composition.items():
            if mono not in self.monomer_smiles: raise ValueError(f"未知单体: {mono}")
            smiles = self.monomer_smiles[mono]
            desc = self.get_descriptors(smiles)
            for k, v in desc.items():
                if mono in hydrophilic: weighted_A[k] += v * num
                elif mono in hydrophobic: weighted_B[k] += v * num

        weighted_A_avg = {f"{k} A avg": v / total_units for k, v in weighted_A.items()}
        weighted_B_avg = {f"{k} B avg": v / total_units for k, v in weighted_B.items()}

        all_features = {}
        all_features['DP A'] = dp_A
        all_features['DP B'] = dp_B
        all_features.update({f"{k} A sum": v for k, v in weighted_A.items()})
        all_features.update({f"{k} B sum": v for k, v in weighted_B.items()})
        all_features.update(weighted_A_avg)
        all_features.update(weighted_B_avg)
        return all_features

    def add_polymer_features_to_df(self, df, polymer_col='StruD'):
        polymer_features_list = []
        for poly_str in df[polymer_col]:
            try:
                feats = self.polymer_features(poly_str)
                polymer_features_list.append(feats)
            except Exception as e:
                print(f"Error: {e}")
                feats_full = {f'{desc} {seg} {typ}': 0 for desc in self.desc_list for seg in ['A','B'] for typ in ['sum','avg']}
                feats_full['DP_A'] = 0; feats_full['DP_B'] = 0
                polymer_features_list.append(feats_full)
        poly_feat_df = pd.DataFrame(polymer_features_list)
        return pd.concat([df.reset_index(drop=True), poly_feat_df], axis=1)

# ================== 2. 辅助函数 ==================
MONOMER_MW = {
    "EG": 44.05, "CL": 114.14, "LA": 72.06, "LLA": 72.06, "DLA": 72.06,
    "GA": 58.04, "PDO": 102.09, "TMC": 102.09, "TOSUO": 172.18, "PG": 74.08, "None": 1.0
}

@st.cache_resource
def load_models():
    try:
        model = joblib.load("XGB_model.joblib")
        preprocessor = joblib.load("preprocessor.joblib")
        return model, preprocessor
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_hts_data():
    """加载高通量筛选生成的 Parquet 文件"""
    file_path = "HTS_Virtual_Results.parquet"
    if os.path.exists(file_path):
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            st.error(f"读取 HTS 数据失败: {e}")
            return None
    else:
        return None

# ================== 3. 页面模式逻辑 ==================
def page_single_prediction():
    """单点实时预测（修复版本，与HTS完全一致的特征对齐流程）"""

    # ====== 加载模型和预处理器 ======
    preprocessor = joblib.load("preprocessor.joblib")
    raw_input_columns = joblib.load("raw_input_columns.joblib")
    post_transform_feature_names = joblib.load("post_transform_feature_names.joblib")
    model_feature_names = joblib.load("model_feature_names.joblib")
    final_feature_list = joblib.load("final_feature_list.joblib")
    model = joblib.load("XGB_model.joblib")

    pf = PolymerFeature()

    st.header("单点实时预测 (Real-time Prediction)")
    st.caption("输入聚合物参数，预测是否形成水凝胶。")

    # ------- 侧边栏输入 -------
    with st.sidebar:
        st.subheader("1. 聚合物结构输入")
        topology = st.selectbox("拓扑结构", ["BAB", "ABA"], index=0)

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            mono_a = st.selectbox("A 单体", ["EG"], index=0)
        with col_a2:
            mn_a_val = st.number_input("A 段分子量", value=1000.0, step=100.0)

        col_b1_1, col_b1_2 = st.columns(2)
        with col_b1_1:
            mono_b1 = st.selectbox("B1 单体", ["CL", "LA", "GA", "PDO", "TOSUO", "TMC"], index=0)
        with col_b1_2:
            mn_b1_val = st.number_input("B1 段分子量", value=700.0, step=100.0)

        col_b2_1, col_b2_2 = st.columns(2)
        with col_b2_1:
            mono_b2 = st.selectbox("B2 单体", ["None", "CL", "LA", "GA", "PDO", "TOSUO", "TMC"], index=0)
        with col_b2_2:
            mn_b2_val = st.number_input("B2 段分子量", value=0.0, step=100.0)

        col_gpc1, col_gpc2 = st.columns(2)
        with col_gpc1:
            gpc = st.number_input("GPC (Mn)", value=2500.0)
        with col_gpc2:
            pdi = st.number_input("PDI", value=1.2)

        # --- 自动计算结构 ---
        dp_a = int(round(mn_a_val / MONOMER_MW.get(mono_a, 100)))
        dp_b1 = int(round(mn_b1_val / MONOMER_MW.get(mono_b1, 100)))
        dp_b2 = int(round(mn_b2_val / MONOMER_MW.get(mono_b2, 100))) if mono_b2 != "None" else 0

        b_part_str = f"({mono_b1}){dp_b1}"
        if mono_b2 != "None" and dp_b2 > 0:
            b_part_str += f"({mono_b2}){dp_b2}"

        a_part_str = f"({mono_a}){dp_a}"

        if topology == "ABA":
            stru_d = f"{a_part_str}{b_part_str}{a_part_str}"
            calc_mn_total = mn_a_val * 2 + mn_b1_val + mn_b2_val
            total_dp_a = dp_a * 2
            total_dp_b = dp_b1 + dp_b2
        else:  # BAB
            stru_d = f"{b_part_str}{a_part_str}{b_part_str}"
            calc_mn_total = mn_a_val + mn_b1_val * 2 + mn_b2_val * 2
            total_dp_a = dp_a
            total_dp_b = (dp_b1 + dp_b2) * 2

        ratio_a = total_dp_a / (total_dp_a + total_dp_b)
        ratio_b = 1 - ratio_a

        st.markdown("---")
        st.code(f"StruD: {stru_d}")


    # ------- 主区输入 -------
    col_main1, col_main2 = st.columns([2, 1])
    with col_main1:
        temperature = st.slider("温度 (°C)", 0.0, 80.0, 37.0)
        concentration = st.slider("浓度 (wt%)", 1.0, 50.0, 20.0)

        # --- 构造输入 DF ---
        df_input = pd.DataFrame({
            'StruD': [stru_d],
            'Topology': [topology],
            'Mn(NMR)': [calc_mn_total],
            'Mn(GPC)': [gpc],
            'PDI': [pdi],
            'Concentration': [concentration],
            'Temperature': [temperature],
            'Ratio A': [ratio_a],
            'Ratio B': [ratio_b],
            'DP A': [total_dp_a],
            'DP B': [total_dp_b]
        })

        if st.button("开始预测", type="primary"):

            # ====== 流程 1: PolymerFeature 特征 ======
            df_features = pf.add_polymer_features_to_df(df_input)

            # ====== 流程 2: 构造 preprocessor 输入列 ======
            # 若缺列 → 填 0
            for col in raw_input_columns:
                if col not in df_features:
                    df_features[col] = 0.0

            X_raw = df_features[raw_input_columns]

            # ====== 流程 3: transform ======
            X_trans = preprocessor.transform(X_raw)
            X_arr = X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans

            X_df = pd.DataFrame(X_arr, columns=post_transform_feature_names)

            # ====== 流程 4: 对齐模型特征 ======
            # 添加缺失列
            for col in model_feature_names:
                if col not in X_df:
                    X_df[col] = 0.0

            # 删除多余列
            extra_cols = [c for c in X_df.columns if c not in model_feature_names]
            X_df.drop(columns=extra_cols, inplace=True)

            # 按顺序重排
            X_final = X_df[model_feature_names]

            # ====== 流程 5: 预测 ======
            prob = model.predict_proba(X_final)[0][1]
            pred = 1 if prob >= 0.5 else 0

            with col_main2:
                if pred == 1:
                    st.success("## Hydrogel")
                else:
                    st.info("## Solution")

                st.metric("Probability", f"{prob*100:.2f}%")

def page_hts_design():
    """新页面：共聚物反向设计 (HTS Explorer)"""
    st.header("共聚物反向设计 (Inverse Design & Screening)")
    st.caption("基于高通量虚拟库，根据目标条件（如体温37°C）筛选最佳单体组合与聚合度。")
    
    df_hts = load_hts_data()
    
    if df_hts is None:
        st.warning("⚠️ 未找到高通量筛选结果文件 `HTS_Virtual_Results.parquet`。请先运行 HTS 脚本生成数据。")
        return

    # --- 侧边栏筛选器 ---
    with st.sidebar:
        st.subheader("1. 设定目标场景")
        
        # 温度选择
        target_temp = st.slider("目标工作温度 (°C)", min_value=0, max_value=80, value=37, step=5)
        
        # 拓扑筛选
        target_topo = st.multiselect("拓扑结构", options=df_hts['Topology'].unique(), default=['ABA'])
        
        # 单体类型筛选
        all_monomers = sorted(df_hts['Hydrophobic_Label'].unique())
        target_monomer = st.multiselect("疏水单体类型 (留空则选所有)", options=all_monomers, default=[])
        
        # 分子量范围
        min_mn, max_mn = int(df_hts['Mn(NMR)'].min()), int(df_hts['Mn(NMR)'].max())
        mn_range = st.slider("Mn(NMR) 范围", min_mn, max_mn, (min_mn, max_mn))

    # --- 数据过滤逻辑 ---
    # 1. 基础条件过滤
    mask = (df_hts['Temperature'] == target_temp) & \
           (df_hts['Topology'].isin(target_topo)) & \
           (df_hts['Mn(NMR)'].between(mn_range[0], mn_range[1]))
    
    if target_monomer:
        mask = mask & (df_hts['Hydrophobic_Label'].isin(target_monomer))
        
    df_filtered = df_hts[mask].copy()

    if df_filtered.empty:
        st.info("当前条件下未找到数据，请放宽筛选条件。")
        return

    st.markdown(f"### ������ 筛选结果 (Temp: {target_temp}°C)")
    
    # --- Tab 分页展示两种排序 ---
    tab1, tab2 = st.tabs(["������ 最佳成胶概率 (Most Stable)", "������ 最低临界凝胶浓度 (Lowest CGC)"])
    
    # 展示列配置
    display_cols = ['StruD', 'Topology', 'Hydrophobic_Label', 'Mn(NMR)', 'Concentration', 'Prob_Gel', 'Pred_Label']
    
    # === Tab 1: 按概率排序 ===
    with tab1:
        st.markdown("**逻辑**：寻找在当前温度和特定浓度下，模型预测为“水凝胶”且置信度最高的结构。")
        
        # 允许用户在这个 Tab 进一步固定浓度
        target_conc = st.selectbox("选择测试浓度 (wt%)", sorted(df_filtered['Concentration'].unique()), index=3) # 默认选个中间的比如20%
        
        df_prob = df_filtered[df_filtered['Concentration'] == target_conc].copy()
        
        # 排序：概率降序
        df_top_prob = df_prob.sort_values(by='Prob_Gel', ascending=False).head(10)
        
        if df_top_prob.empty:
            st.warning(f"在 {target_temp}°C, {target_conc} wt% 下没有找到任何数据。")
        else:
            st.dataframe(
                df_top_prob[display_cols],
                column_config={
                    "Prob_Gel": st.column_config.ProgressColumn(
                        "Gel Probability",
                        help="成胶概率",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "StruD": st.column_config.TextColumn("Structure (Detail)", width="medium"),
                },
                hide_index=True,
                use_container_width=True
            )
            st.success(f"������ 推荐：{df_top_prob.iloc[0]['StruD']} (Prob: {df_top_prob.iloc[0]['Prob_Gel']:.2f})")

    # === Tab 2: 按 CGC 排序 (Lowest CGC) ===
    with tab2:
        st.markdown("**逻辑**：寻找在当前温度下，能在**最低浓度**形成水凝胶的结构 (高效能材料)。")
        
        # 1. 只看预测为 Gel 的数据
        df_gels = df_filtered[df_filtered['Pred_State'] == 1].copy()
        
        if df_gels.empty:
            st.warning(f"在 {target_temp}°C 下，筛选范围内没有预测出任何水凝胶。")
        else:
            # 2. 寻找每个唯一结构 (StruD) 的最低浓度
            # 按照 ID_Virtual 或 StruD 分组，取 Concentration 的最小值
            # 为了展示信息，我们先按 StruD 分组找到最小 Conc，然后 merge 回去取详情
            
            # 方法：按 Concentration 升序排序，然后去重 StruD，保留第一次出现的（即最小浓度）
            df_cgc = df_gels.sort_values(by=['Concentration', 'Prob_Gel'], ascending=[True, False])
            df_best_cgc = df_cgc.drop_duplicates(subset=['StruD'], keep='first').head(10)
            
            st.dataframe(
                df_best_cgc[display_cols],
                column_config={
                    "Concentration": st.column_config.NumberColumn(
                        "Min Concentration (CGC)",
                        help="在该温度下成胶的最低浓度",
                        format="%d wt%"
                    ),
                    "Prob_Gel": st.column_config.NumberColumn("Prob at CGC", format="%.2f"),
                    "StruD": st.column_config.TextColumn("Structure (Detail)", width="medium"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            if not df_best_cgc.empty:
                best_poly = df_best_cgc.iloc[0]
                st.success(f"������ CGC 冠军：{best_poly['StruD']} \n\n 在 {best_poly['Concentration']} wt% 即可成胶 (Prob: {best_poly['Prob_Gel']:.2f})")

# ================== 4. 主程序入口 ==================

def main():
    st.sidebar.title("功能导航")
    app_mode = st.sidebar.radio("选择模式", ["������ 单点实时预测", "������ 共聚物反向设计 (HTS)"])
    st.sidebar.markdown("---")

    if app_mode == "������ 单点实时预测":
        page_single_prediction()
    elif app_mode == "������ 共聚物反向设计 (HTS)":
        page_hts_design()

if __name__ == "__main__":
    main()