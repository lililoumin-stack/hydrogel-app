import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

# ================== 1. 核心特征工程类 (保持不变) ==================
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

class PolymerFeature:
    """计算聚合物特征（支持单个/批量处理，返回加权和 & 加权平均值）"""

    def __init__(self):
        self.carbonyl_pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
        # 增强描述符列表（保持和你主程序需要的列表一致）
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
    #匹配 C=O 双键    
    def get_carbonyl_count(mol):
        pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
        return len(mol.GetSubstructMatches(pattern))
    
    def get_descriptors(self, smiles):
        """提取单体分子描述符（增强版）"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {k: 0 for k in self.desc_list}

        desc = {
            # ====== 你原本的 ======
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),

            # ====== 原子层面基础结构特征 ======
            "MW": Descriptors.MolWt(mol),
            "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
            "RotatableBonds": Lipinski.NumRotatableBonds(mol),
            "HeteroatomCount": rdMolDescriptors.CalcNumHeteroatoms(mol),

            # ====== 芳香性、环结构 ======
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "AliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "SaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
            "AromaticAtoms": sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),

            # ====== 电子性质（charge / polar） ======
            "FormalCharge": Chem.GetFormalCharge(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),

            # ====== 子结构性质（极性、键类型） ======
            "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),
            "StereoCenterCount": rdMolDescriptors.CalcNumAtomStereoCenters(mol), 
            "CarbonylCount": len(mol.GetSubstructMatches(self.carbonyl_pattern)),

            # ====== 表面积相关 ======
            "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),

            # ====== 环结构细节 ======
            "NumRingsSharingAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            "NumBicyclicAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
            
            # 补充列表里有但此处未计算的，防止 Key Error（设为0）
            "VSA_Estate1": 0,
            "VSA_Estate2": 0,
        }

        # 处理 None 或 NAN
        desc = {k: (v if v is not None else 0) for k, v in desc.items()}
        
        return desc

    def parse_polymer(self, polymer_str):
        """
        解析类似 (CL)5.9(TMC)2.4-(EG)22.7 这种字符串
        【修复】: 现在支持重复单体累加 (如 ABA 结构 (CL)10(EG)20(CL)10)，防止后面的值覆盖前面的值
        """
        pattern = r"\((.*?)\)([\d\.]+)"
        matches = re.findall(pattern, polymer_str)
        
        composition = {}
        for monomer, num_str in matches:
            num = float(num_str)
            if monomer in composition:
                composition[monomer] += num  # 如果单体已存在，累加聚合度
            else:
                composition[monomer] = num   # 否则新建
        
        return composition

    def polymer_features(self, polymer_str):
        """计算聚合物亲水段与疏水段的特征（加权和 + 加权平均）"""
        # 单体分类
        hydrophilic = {'EG'}
        hydrophobic = {'CL','LA','LLA','DLA','TMC','GA','PDO','TOSUO','PG'}

        composition = self.parse_polymer(polymer_str)
        total_units = sum(composition.values())
        if total_units == 0:
             raise ValueError("总聚合度为0")

        # 计算亲水段 (A) 和疏水段 (B) 的单体数量之和
        dp_A = sum(num for mono, num in composition.items() if mono in hydrophilic)
        dp_B = sum(num for mono, num in composition.items() if mono in hydrophobic)

        # 亲水段(A)和疏水段(B)特征存储
        weighted_A = {}  # hydrophilic
        weighted_B = {}  # hydrophobic

        # 初始化字典，防止只有A或只有B时报错
        sample_desc = self.get_descriptors(self.monomer_smiles['EG']) #以此为模板
        for k in sample_desc.keys():
            weighted_A[k] = 0
            weighted_B[k] = 0

        # 遍历并计算权重累积
        for mono, num in composition.items():
            if mono not in self.monomer_smiles:
                raise ValueError(f"未知单体: {mono}")
            
            smiles = self.monomer_smiles[mono]
            desc = self.get_descriptors(smiles)

            for k, v in desc.items():
                if mono in hydrophilic:  # 若为亲水段
                    weighted_A[k] += v * num
                elif mono in hydrophobic:  # 若为疏水段
                    weighted_B[k] += v * num
                # else: 已在parse校验，此处通常不会触发

        # 计算加权平均（除以总单体数）
        weighted_A_avg = {f"{k} A avg": v / total_units for k, v in weighted_A.items()}
        weighted_B_avg = {f"{k} B avg": v / total_units for k, v in weighted_B.items()}

        # 总输出整理
        all_features = {}
        all_features['DP A'] = dp_A
        all_features['DP B'] = dp_B
        # 加入 sum 特征
        all_features.update({f"{k} A sum": v for k, v in weighted_A.items()})
        all_features.update({f"{k} B sum": v for k, v in weighted_B.items()})

        # 加入 avg 特征
        all_features.update(weighted_A_avg)
        all_features.update(weighted_B_avg)

        return all_features

    def add_polymer_features_to_df(self, df, polymer_col='StruD'):
        """批量处理 DataFrame，添加聚合物增强描述符特征"""
        polymer_features_list = []
        for poly_str in df[polymer_col]:
            try:
                feats = self.polymer_features(poly_str)
                polymer_features_list.append(feats)
            except Exception as e:
                print(f"处理聚合物 {poly_str} 时出错: {e}")
                feats_full = {f'{desc} {seg} {typ}': 0 
                            for desc in self.desc_list
                            for seg in ['A','B']
                            for typ in ['sum','avg']}
                feats_full['DP_A'] = 0
                feats_full['DP_B'] = 0
                polymer_features_list.append(feats_full)
            

        poly_feat_df = pd.DataFrame(polymer_features_list)
        df_with_poly = pd.concat([df.reset_index(drop=True), poly_feat_df], axis=1)
        return df_with_poly

# ================== 2. Streamlit 页面逻辑 ==================
st.set_page_config(page_title="AI 水凝胶预测系统", layout="wide")

# 辅助数据：单体重复单元的近似分子量 (用于将 Mn 转换为 聚合度 DP)
# 注意：这里使用常见重复单元的分子量，如有偏差可在此修正
MONOMER_MW = {
    "EG": 44.05,
    "CL": 114.14,
    "LA": 72.06,
    "LLA": 72.06,
    "DLA": 72.06,
    "GA": 58.04,
    "PDO": 102.09,
    "TMC": 102.09,
    "TOSUO": 172.18,
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
        mn_a_val = st.number_input("分子量", value=1000.0, step=100.0)

    # 3. B1 嵌段输入
    col_b1_1, col_b1_2 = st.columns(2)
    with col_b1_1:
        mono_b1 = st.selectbox("B1 单体", ["CL", "LA", "GA", "PDO", "TOSUO", "TMC"], index=0)
    with col_b1_2:
        mn_b1_val = st.number_input("分子量", value=700.0, step=100.0)

    # 4. B2 嵌段输入
    col_b2_1, col_b2_2 = st.columns(2)
    with col_b2_1:
        mono_b2 = st.selectbox("B2 单体", ["None", "CL", "LA", "GA", "PDO", "TOSUO", "TMC"], index=0)
    with col_b2_2:
        mn_b2_val = st.number_input("分子量", value=0.0, step=100.0)

    # 5. GPC 和 PDI
    col_gpc1, col_gpc2 = st.columns(2)
    with col_gpc1:
        gpc = st.number_input("GPC (Mn)", value=2500.0)
    with col_gpc2:
        pdi = st.number_input("PDI", value=1.2)

    # --- 自动计算逻辑 ---
    # 1. 计算单段聚合度 (DP) = Mn / 单体分子量
    dp_a = int(round(mn_a_val / MONOMER_MW.get(mono_a, 100)))
    dp_b1 = int(round(mn_b1_val / MONOMER_MW.get(mono_b1, 100)))
    dp_b2 = int(round(mn_b2_val / MONOMER_MW.get(mono_b2, 100))) if mono_b2 != "None" else 0

    # 2. 生成 StruD 字符串 & 计算总聚合度 (Total DP)
    # 构建各部分的字符串片段
    b_part_str = f"({mono_b1}){dp_b1}"
    if mono_b2 != "None" and dp_b2 > 0:
        b_part_str += f"({mono_b2}){dp_b2}"
    
    a_part_str = f"({mono_a}){dp_a}"

    # 初始化变量，防止 NameError
    total_dp_a = 0
    total_dp_b = 0

    if topology == "ABA":
        # 结构: A(a) - B1(b1)B2(b2) - A(a)
        stru_d = f"{a_part_str}{b_part_str}{a_part_str}"
        
        # Mn 计算 (保持质量守恒: 两端 A + 中间 B1+B2)
        calc_mn_total = (mn_a_val * 2) + mn_b1_val + mn_b2_val
        
        # 聚合度计算
        total_dp_a = dp_a * 2
        total_dp_b = dp_b1 + dp_b2

    else: # BAB
        # 结构: B1(b1)B2(b2) - A(a) - B1(b1)B2(b2)
        stru_d = f"{b_part_str}{a_part_str}{b_part_str}"
        
        # Mn 计算 (两端 B1+B2 + 中间 A)
        calc_mn_total = mn_a_val + (mn_b1_val * 2) + (mn_b2_val * 2)
        
        # 聚合度计算
        total_dp_a = dp_a
        total_dp_b = (dp_b1 + dp_b2) * 2

    # 3. 计算摩尔比 (Ratio)
    total_dp = total_dp_a + total_dp_b
    
    calc_ratio_a = total_dp_a / total_dp if total_dp > 0 else 0
    calc_ratio_b = total_dp_b / total_dp if total_dp > 0 else 0

    # 显示计算结果预览
    st.markdown("---")
    st.markdown("**结构参数:**")
    st.code(f"StruD: {stru_d}", language="text")
    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Mn", f"{calc_mn_total:.1f}")
    col_res2.metric("Ratio A", f"{calc_ratio_a:.3f}")
    col_res3.metric("Ratio B", f"{calc_ratio_b:.3f}")
    calc_ratio_b = 1.0 - calc_ratio_a

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
        'Mn(NMR)': [calc_mn_total], # 使用计算出的总 Mn
        'Mn(GPC)': [gpc],
        'PDI': [pdi],
        'Concentration': [concentration],
        'Temperature': [temperature],
        'Ratio A': [calc_ratio_a], # 使用计算出的 Ratio_A
        'Ratio B': [calc_ratio_b],  # 使用计算出的 Ratio_B
        'DP A': [total_dp_a], 
        'DP B': [total_dp_b]
    }
    df_input = pd.DataFrame(input_data)

    if st.button("开始预测 (Predict)", type="primary"):
        if model and preprocessor:
            try:
                # 1. 特征工程
                df_features = pf.add_polymer_features_to_df(df_input)
                
                # 2. 构造完整的特征列表 (对应 Preprocessor 的输入)
                # 注意：这里只列出 Preprocessor 需要处理的原始列
                base_features = [
                                    'Topology', 
                                    'Mn(NMR)', 
                                    'Mn(GPC)', 'PDI',
                                    'Concentration', 'Temperature',
                                    'Ratio A','Ratio B',
                                    'DP A','DP B',
                                ]
                # 动态获取生成的聚合物特征列 (现在的列名已经是 "LogP A sum" 格式了)
                poly_cols = [c for c in df_features.columns if c.endswith(' sum')]
                features_for_preprocessor = base_features + poly_cols
                
                # 提取用于预处理的 X
                X_raw = df_features[features_for_preprocessor]
                
                # 3. 预处理 (Transform)
                # 这会生成包含所有特征(未筛选)的矩阵
                X_processed_array = preprocessor.transform(X_raw)
                
                # 4. --- 关键修复：特征对齐 (Feature Alignment) ---
                
                # A. 重建预处理后的列名 (为了能按名字索引)
                # 获取 OneHot 后的列名
                try:
                    cat_cols = ['Topology']
                    num_cols = [c for c in features_for_preprocessor if c not in cat_cols]
                    ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
                    all_processed_cols = num_cols + list(ohe_cols)
                except Exception as e:
                    st.error(f"预处理列名生成失败: {e}")
                    st.stop()
                
                # 转为 DataFrame
                X_processed_df = pd.DataFrame(X_processed_array, columns=all_processed_cols)
                
                # B. 获取模型真正需要的特征
                # XGBoost 模型对象通常保存了 feature_names_in_
                try:
                    if hasattr(model, 'feature_names_in_'):
                        model_features = model.feature_names_in_
                    else:
                        # 如果版本旧，尝试用 get_booster
                        model_features = model.get_booster().feature_names
                    
                    # C. 只保留模型需要的列 (自动剔除训练时drop掉的列)
                    X_final = X_processed_df[model_features]
                    
                except KeyError as e:
                    st.error(f"特征缺失错误: 模型需要 {e}，但预处理结果中没有。请检查列名格式。")
                    st.write("当前生成的列名:", all_processed_cols)
                    st.stop()
                except Exception as e:
                    st.error(f"无法获取模型特征列表，尝试直接预测可能失败: {e}")
                    X_final = X_processed_df # 盲目尝试
                
                # 5. 预测
                prediction = model.predict(X_final)[0]
                probability = model.predict_proba(X_final)[0]

                # ... (后续显示结果的代码不变) ...
                #显示结果
                with col_main2:
                    st.markdown("### 预测结果")
                    if prediction == 1:
                        st.success("## Hydrogel (水凝胶)")
                        st.metric("Confidence(置信度)", f"{probability[1]*100:.2f}%")
                    else:
                        st.info("## Solution (溶液)")
                        st.metric("Confidence(置信度)", f"{probability[0]*100:.2f}%")
                    
                    st.markdown("---")
                    #st.write("**输入特征快照:**")
                    #st.json(input_data)

            except Exception as e:
                st.error(f"预测出错: {str(e)}\n请检查特征列是否与模型匹配。")
        else:
            st.error("模型未加载，无法预测。")

# 添加一个提示框
st.markdown("---")
st.caption("小贴士: 在左侧修改单体类型和分子量，系统会自动计算聚合度并生成结构式。")
