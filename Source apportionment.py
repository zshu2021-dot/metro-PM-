# ML_for_metro_PM.py

import os
import pandas as pd
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint, uniform
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from xgboost import XGBRegressor
import shap
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYES_AVAILABLE = True
    print("✓ BayesSearchCV 可用（贝叶斯优化）")
except ImportError:
    BAYES_AVAILABLE = False
    print("⚠ BayesSearchCV 不可用，回退到 RandomizedSearchCV")
    print("  安装命令: pip install scikit-optimize")

# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("提示: 未安装 LightGBM，将跳过该模型 (安装: pip install lightgbm)")

# ==================== 输出目录配置 ====================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out_path(filename):
    """返回输出文件的完整路径"""
    return os.path.join(OUTPUT_DIR, filename)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50

# 设置随机种子
np.random.seed(42)

print("=" * 100)
print("地铁颗粒物浓度预测分析系统")
print("=" * 100)

# ==================== 1. 数据加载和预处理（增强版） ====================
def load_and_preprocess_data(file_path):
    """加载并预处理数据（修正变量定义顺序）"""
    print("\n" + "=" * 100)
    print("步骤 1: 数据加载和预处理")
    print("=" * 100)

    df = pd.read_excel(file_path)
    feature_cols = ['Peak', 'Platform depth', 'Metro humidity', 
                   'Outdoor humidity', 'Metro temperature', 'Outdoor temperature',
                   'Platform years',  'Screen door type',
                   'Platform type', 'Transfer station',
                   'Ground_PM']
    target_col = 'Metro_PM'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # 1. 缺失值处理
    if X.isnull().sum().sum() > 0:
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
    if y.isnull().sum() > 0:
        y.fillna(y.median(), inplace=True)
        
    # 2. 异常值截断 (IQR方法)
    continuous_cols = ['Platform depth', 'Metro humidity', 'Outdoor humidity',
                       'Metro temperature', 'Outdoor temperature', 'Platform years',
                       'Line years', 'Air pressure', 'Ground_PM']
    for col in continuous_cols:
        if col in X.columns:
            Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            IQR = Q3 - Q1
            X[col] = X[col].clip(Q1 - 3.0 * IQR, Q3 + 3.0 * IQR)
    
    # 3. 构建温差特征并精简特征列
    print("构建物理特征并精简变量...")
    X['Temp_diff'] = X['Metro temperature'] - X['Outdoor temperature']

    # 定义最终保留的纯净特征（不含冗余室外温湿和气压）
    final_keep_cols = [
        'Ground_PM',  'Metro humidity','Metro temperature', 'Screen doort ype',
        'Temp_diff',  'Peak', 'Transfer station', 'Platform depth'
    ]
    
    # 确保只保留选定的列
    X = X[final_keep_cols]
    all_feature_cols = final_keep_cols
    
     # ✅ 使用分层抽样，确保训练集和测试集的y分布一致
    print("正在划分数据集（分层抽样）...")
    y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')  # 4分位数分箱
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y_bins   # ← 核心修改：按y的分布分层
    )
    
    # 诊断打印（必须在split之后）
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train: 均值={y_train.mean():.2f}, std={y_train.std():.2f}, "
          f"范围=[{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"y_test:  均值={y_test.mean():.2f}, std={y_test.std():.2f}, "
          f"范围=[{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # 标准化
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"数据处理完成。最终特征数: {len(all_feature_cols)}")
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, all_feature_cols


# ==================== 残差驱动数据增强（回归专用） ====================
def residual_based_augmentation(X, y, n_aug=1, noise_scale=0.25):

    # ✅ 1. 提前初始化所有关键变量，解决Pylance未定义报错
    sigma = 1.0
    X_aug_list = []  # 提前声明空列表
    y_aug_list = []  # 提前声明空列表

    # 2. 拟合基准线性模型（小样本用更强正则，避免基准模型过拟合）
    base_model = Ridge(alpha=100.0)  # 小样本增强正则
    base_model.fit(X, y)

    y_pred = base_model.predict(X)
    residuals = y - y_pred

    # 3. 重新赋值残差分布参数（覆盖初始值，逻辑不变）
    mu = 0  # 小样本残差均值置0，避免系统偏差
    sigma = residuals.std()

    # 4. 定义连续特征列（仅对这些特征加微小噪声，分类特征不变）
    continuous_cols = ['Platform depth', 'Metro humidity', 'Outdoor humidity', 
                       'Metro temperature', 'Outdoor temperature', 
                       'Air pressure', 'Ground_PM']
    # 提高特征噪声的精细度，分特征设置不同噪声强度（贴合业务实际，避免关键特征过度扰动）
    feat_noise_scale_map = {
        'Platform depth': 0.005, 'Metro humidity': 0.01, 'Outdoor humidity': 0.01,
        'Metro temperature': 0.01, 'Outdoor temperature': 0.01,
        'Air pressure': 0.01, 'Ground_PM': 0.03
    }

    # 5. 生成增强样本
    for _ in range(n_aug):
        # 标签噪声：进一步降低强度+更严格的值域约束（原1.1→1.05，避免生成不合理的PM值）
        noise = np.random.normal(mu, sigma * noise_scale, size=len(y))
        y_new = y_pred + noise
        y_new = np.clip(y_new, 0, y.max() * 1.05)  # 上限不超过原最大值的105%
        
        # 特征噪声：按特征个性化设置噪声强度，而非统一0.01
        X_new = X.copy()
        for col in continuous_cols:
            if col in X_new.columns:
                feat_noise = np.random.normal(0, X_new[col].std() * feat_noise_scale_map[col], size=len(X_new))
                X_new[col] += feat_noise
                # 更严格的业务值域约束，避免生成不合理特征值
                if 'humidity' in col.lower():
                    X_new[col] = np.clip(X_new[col], 20, 95)  
                elif 'temperature' in col.lower():
                    X_new[col] = np.clip(X_new[col], 10, 35) 
                elif 'Platform depth' in col:
                    X_new[col] = np.clip(X_new[col], 5, 25)  
                elif 'PM' in col:
                    X_new[col] = np.clip(X_new[col], 0, 300)  
                elif 'Air pressure' in col:
                    X_new[col] = np.clip(X_new[col], 980, 1050) 
        
        categorical_cols = ['Peak', 'Transfer station']  # 离散分类特征
        for col in categorical_cols:
            if col in X_new.columns:
                # 5%的概率翻转分类值，避免分类特征完全不变
                flip_mask = np.random.choice([True, False], size=len(X_new), p=[0.05, 0.95])
                X_new.loc[flip_mask, col] = 1 - X_new.loc[flip_mask, col]
        
        X_aug_list.append(X_new)
        y_aug_list.append(pd.Series(y_new, index=y.index))
    
    X_aug = pd.concat(X_aug_list, axis=0)
    y_aug = pd.concat(y_aug_list, axis=0)
    return X_aug, y_aug

# ==================== 2. Adaptive LASSO实现 ====================
from sklearn.base import BaseEstimator, RegressorMixin
class AdaptiveLasso(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"  
    """Adaptive LASSO回归实现"""
    def __init__(self, alpha=1.0, gamma=2.0, max_iter=10000):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        # 第一步：使用Ridge回归获得初始权重
        ridge = Ridge(alpha=0.001)
        ridge.fit(X, y)
        ridge_coef = np.abs(ridge.coef_)
        
        # 计算自适应权重
        ridge_coef_clipped = np.clip(np.abs(ridge.coef_), 1e-4, None)  # ↑ 设置下限
        weights = 1.0 / (ridge_coef_clipped ** self.gamma)
        weights = weights / weights.sum() * len(weights)  # ↑ 归一化权重
        
        # 第二步：使用加权的LASSO
        X_weighted = X * weights
        
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        lasso.fit(X_weighted, y)
        
        self.coef_ = lasso.coef_ * weights
        self.intercept_ = lasso.intercept_
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    def get_params(self, deep=True):
        return {'alpha': self.alpha, 'gamma': self.gamma, 'max_iter': self.max_iter}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# ==================== 3. 模型定义（贝叶斯搜索空间版）====================
def get_models_and_params(n_train=360):
    if BAYES_AVAILABLE:
        models = {
            # ── 线性模型：维持当前稳定参数 ──────────────────────────────
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': Real(0.001, 10.0, prior='log-uniform'),
                },
                'use_scaled': True, 'n_iter': 30
            },
            'Lasso': {
                'model': Lasso(max_iter=500000),
                'params': {
                    'alpha': Real(0.0001, 0.1, prior='log-uniform'),
                },
                'use_scaled': True, 'n_iter': 30
            },
            'ElasticNet': {
                'model': ElasticNet(max_iter=500000, tol=1e-5),
                'params': {
                    'alpha':    Real(0.0001, 0.1, prior='log-uniform'),
                    'l1_ratio': Real(0.1, 0.9, prior='uniform'),
                },
                'use_scaled': True, 'n_iter': 30
            },

            'Adaptive Lasso': {
                'model': AdaptiveLasso(),
                'params': {
                    'alpha': Real(0.00001, 0.1, prior='log-uniform'),
                    'gamma': Real(0.1, 0.9, prior='uniform'),
                },
                'use_scaled': True, 'n_iter': 30
            },

            # ── Extra Trees：修复报错，调整为最强基模型 ────────────────
            'Extra Trees': {
                # ✅ 修复报错：明确设置 bootstrap=True
                'model': ExtraTreesRegressor(random_state=42, n_jobs=-1, bootstrap=True),
                'params': {
                    'n_estimators':      Integer(200, 500),
                    'max_depth':         Integer(3, 5),          
                    'min_samples_split': Integer(15, 40),
                    'min_samples_leaf':  Integer(10, 25),
                    'max_features':      Real(0.5, 0.85, prior='uniform'),
                    'max_samples':       Real(0.5, 0.8, prior='uniform'),
                },
                'use_scaled': False, 'n_iter': 50
            },

            # ── Random Forest：放宽限制，恢复拟合能力 ──────────────────
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True),
                'params': {
                    'n_estimators':      Integer(100, 300),
                    'max_depth':         Integer(3, 5),          
                    'min_samples_split': Integer(10, 25),          
                    'min_samples_leaf':  Integer(15, 30),          
                    'max_features':      Real(0.3, 0.6, prior='uniform'),
                    'max_samples':       Real(0.4, 0.65, prior='uniform'),
                },
                'use_scaled': False, 'n_iter': 40
            },

            # ── XGBoost：平滑学习，加强泛化 ──────────────────────────
            'XGBoost': {
                'model': XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
                'params': {
                    'n_estimators':      Integer(30, 80),
                    'max_depth':         Integer(1, 3),
                    'learning_rate':     Real(0.01, 0.1, prior='log-uniform'),
                    'subsample':         Real(0.3, 0.6, prior='uniform'),
                    'colsample_bytree':  Real(0.3, 0.6, prior='uniform'),
                    'reg_alpha':         Real(100, 500.0, prior='log-uniform'),
                    'reg_lambda':        Real(100, 500.0, prior='log-uniform'),
                    'min_child_weight':  Integer(15, 40),
                    'gamma':             Real(1.0, 5.0),
                },
                'use_scaled': False, 'n_iter': 40
            },

            # ── Gradient Boosting：加大叶节点约束 ──────────────────────
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators':        Integer(30, 80),
                    'max_depth':           Integer(1, 2),
                    'learning_rate':       Real(0.01, 0.1, prior='log-uniform'),
                    'subsample':           Real(0.3, 0.7, prior='uniform'),
                    'min_samples_leaf':    Integer(20, 50),      
                    'max_features':        Real(0.3, 0.7, prior='uniform'),
                    'validation_fraction': Real(0.15, 0.2),
                    'n_iter_no_change':    Integer(5, 10),
                    'tol':                 Real(1e-4, 1e-3),
                },
                'use_scaled': False, 'n_iter': 40
            },

            # ── KNN ──────────────────────────
            'KNN': {
                'model': KNeighborsRegressor(n_jobs=-1),
                'params': {
                    'n_neighbors': Integer(40, 80),
                    'weights':     Categorical(['uniform']),
                    'metric':      Categorical(['euclidean', 'manhattan']),
                    'leaf_size':   Integer(20, 50),
                },
                'use_scaled': True, 'n_iter': 20
            },

            # ── SVM：参数回归平稳区间 ────────────────────────────────
            'SVM': {
                'model': SVR(),
                'params': {
                    'C':       Real(10, 200.0, prior='log-uniform'),
                    'epsilon': Real(0.05, 0.3, prior='uniform'),
                    'gamma':   Categorical(['scale']),
                    'kernel':  Categorical(['rbf']),
                },
                'use_scaled': True, 'n_iter': 25
            },
        }

        if LGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators':      Integer(30, 80),
                    'max_depth':         Integer(1, 2),
                    'learning_rate':     Real(0.01, 0.08, prior='log-uniform'),
                    'num_leaves':        Integer(2, 4),
                    'subsample':         Real(0.3, 0.6, prior='uniform'),
                    'colsample_bytree':  Real(0.3, 0.6, prior='uniform'),
                    'reg_alpha':         Real(20.0, 100.0, prior='log-uniform'),
                    'reg_lambda':        Real(20.0, 100.0, prior='log-uniform'),
                    'min_child_samples': Integer(45, 80),
                    'min_gain_to_split': Real(2.0, 8.0),
                },
                'use_scaled': False, 'n_iter': 40
            }

    else:
        # 随机搜索兜底方案（与上面对齐）
        models = {
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.0001, 0.001, 0.01, 0.1]},
                'use_scaled': True, 'n_iter': 15
            },
            'Lasso': {
                'model': Lasso(max_iter=200000),
                'params': {'alpha': [0.00001, 0.0001, 0.001, 0.01]},
                'use_scaled': True, 'n_iter': 15
            },
            'ElasticNet': {
                'model': ElasticNet(max_iter=200000),
                'params': {'alpha': [0.00001, 0.0001, 0.001], 'l1_ratio': [0.1, 0.3, 0.5, 0.8]},
                'use_scaled': True, 'n_iter': 15
            },
            'Adaptive Lasso': {
                'model': AdaptiveLasso(),
                'params': {'alpha': [0.00001, 0.0001, 0.001], 'gamma': [0.2, 0.4, 0.6, 0.8]},
                'use_scaled': True, 'n_iter': 15
            },
            'Extra Trees': {
                'model': ExtraTreesRegressor(random_state=42, n_jobs=-1, bootstrap=True),
                'params': {
                    'n_estimators': [200, 300, 400], 'max_depth': [5, 6, 7, 8],
                    'min_samples_leaf': [4, 6, 8], 'min_samples_split': [4, 8, 12],
                    'max_features': [0.5, 0.6, 0.7, 0.8], 'max_samples': [0.5, 0.6, 0.7]
                },
                'use_scaled': False, 'n_iter': 25
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [200, 300, 400], 'max_depth': [4, 5, 6],
                    'min_samples_leaf': [5, 8, 12], 'min_samples_split': [5, 10, 15],
                    'max_features': [0.5, 0.6, 0.7], 'max_samples': [0.5, 0.6, 0.7]
                },
                'use_scaled': False, 'n_iter': 25
            },
            'XGBoost': {
                'model': XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
                'params': {
                    'n_estimators': [40, 60, 80, 100], 'max_depth': [2, 3, 4],
                    'learning_rate': [0.02, 0.05, 0.1], 'reg_alpha': [5.0, 20.0, 50.0],
                    'reg_lambda': [10.0, 50.0, 100.0], 'min_child_weight': [10, 20, 30],
                    'gamma': [0.5, 2.0, 5.0], 'subsample': [0.5, 0.7], 'colsample_bytree': [0.5, 0.7]
                },
                'use_scaled': False, 'n_iter': 25
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150], 'max_depth': [2, 3],
                    'learning_rate': [0.02, 0.05, 0.1], 'min_samples_leaf': [15, 25, 35],
                    'subsample': [0.5, 0.7], 'max_features': [0.5, 0.7]
                },
                'use_scaled': False, 'n_iter': 20
            },
            'KNN': {
                'model': KNeighborsRegressor(n_jobs=-1),
                'params': {
                    'n_neighbors': [10, 15, 20, 25, 30],
                    'weights': ['uniform'],  
                    'metric': ['euclidean', 'manhattan']
                },
                'use_scaled': True, 'n_iter': 10
            },
            'SVM': {
                'model': SVR(),
                'params': {
                    'C': [10.0, 50.0, 100.0, 200.0], 'epsilon': [0.05, 0.1, 0.2],
                    'gamma': ['scale'], 'kernel': ['rbf']
                },
                'use_scaled': True, 'n_iter': 15
            },
        }

        if LGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 150], 'max_depth': [3, 4, 5],
                    'learning_rate': [0.02, 0.05, 0.1], 'num_leaves': [12, 20, 31],
                    'reg_alpha': [5.0, 20.0, 50.0], 'reg_lambda': [10.0, 50.0, 100.0],
                    'min_child_samples': [15, 25, 40], 'subsample': [0.5, 0.7],
                    'colsample_bytree': [0.5, 0.7]
                },
                'use_scaled': False, 'n_iter': 25
            }

    return models

# ==================== 4. 搜索器构建函数====================
def build_search(model, params, n_iter, n_cv):
    """
    根据可用库自动选择最优搜索策略：
    贝叶斯优化（BayesSearchCV）> 随机搜索（RandomizedSearchCV）

    返回：(search对象, 搜索方式描述字符串)
    """
    if BAYES_AVAILABLE:
        try:
            search = BayesSearchCV(
                estimator   = model,
                search_spaces = params,
                n_iter      = n_iter,
                cv          = n_cv,
                scoring     = 'r2',
                n_jobs      = -1,
                random_state= 42,
                verbose     = 0,
                refit       = True,
                return_train_score = True,
                optimizer_kwargs   = {'base_estimator': 'GP',  # 高斯过程代理模型
                                      'acq_func': 'EI'}        # 期望提升采集函数
            )
            return search, "贝叶斯优化（BayesSearchCV·GP·EI）"
        except Exception as e:
            print(f"  ⚠ BayesSearchCV初始化失败（{e}），回退随机搜索")

    # 兜底：随机搜索
    search = RandomizedSearchCV(
        estimator   = model,
        param_distributions = params,
        n_iter      = n_iter,
        cv          = n_cv,
        scoring     = 'r2',
        n_jobs      = -1,
        random_state= 42,
        verbose     = 0,
        refit       = True,
        error_score = 'raise',
        return_train_score = True
    )
    return search, "随机搜索（RandomizedSearchCV）"


# ==================== 4. 模型训练和评估====================
def train_and_evaluate_models(X_train, X_test, y_train, y_test,
                               X_train_scaled, X_test_scaled):
    n_train = len(X_train)
    n_cv    = 10
    models_config = get_models_and_params(n_train=n_train)
    results        = {}
    trained_models = {}

    print("\n" + "=" * 100)
    print(f"步骤 2: 模型训练与评估（共 {len(models_config)} 个模型）")
    print("=" * 100)

    for idx, (model_name, config) in enumerate(models_config.items(), 1):
        print(f"\n[{idx}/{len(models_config)}] {model_name}")
        X_tr, X_te = (X_train_scaled, X_test_scaled) \
                     if config['use_scaled'] else (X_train, X_test)
            
        # ✅ 修复1：SVM单独使用StandardScaler
        if model_name == 'SVM':
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler()
            X_tr = pd.DataFrame(
                ss.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_te = pd.DataFrame(
                ss.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

        try:
            n_iter = config.get('n_iter', 40)
            search, search_desc = build_search(
                config['model'], config['params'], n_iter, n_cv)

            # ✅ 所有模型统一用同一方式fit，不在循环外重复
            search.fit(X_tr, y_train)

            best_model  = search.best_estimator_
            best_params = search.best_params_

            if model_name == 'KNN':
                current_k = getattr(best_model, 'n_neighbors', 1)
                if current_k < 15:
                    print(f"  ⚠ KNN n_neighbors={current_k} 过小，强制重设为15")
                    best_model.set_params(n_neighbors=15)
                    best_model.fit(X_tr, y_train)

            y_pred_train = best_model.predict(X_tr)
            y_pred_test  = best_model.predict(X_te)

            train_r2   = r2_score(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_mae  = mean_absolute_error(y_train, y_pred_train)

            # ✅ 使用 RepeatedKFold 确保评估稳定
            rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
            cv_scores = cross_val_score(
                best_model, X_tr, y_train,
                cv=rkf, scoring='r2', n_jobs=-1)

            test_r2   = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae  = mean_absolute_error(y_test, y_pred_test)

            mask      = y_test != 0
            test_mape = np.mean(
                np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])
            ) * 100
            gap = train_r2 - test_r2

            results[model_name] = {
                'best_params':  dict(best_params),
                'train_r2':     train_r2,
                'test_r2':      test_r2,
                'train_rmse':   train_rmse,
                'test_rmse':    test_rmse,
                'train_mae':    train_mae,
                'test_mae':     test_mae,
                'test_mape':    test_mape,
                'cv_r2_mean':   cv_scores.mean(),
                'cv_r2_std':    cv_scores.std(),
                'overfit_gap':  gap,
                'y_pred_train': y_pred_train,
                'y_pred_test':  y_pred_test,
                'search_mode':  search_desc
            }
            trained_models[model_name] = {
                'model':      best_model,
                'use_scaled': config['use_scaled']
            }

            print(f"  Test R²={test_r2:.4f} | "
                  f"CV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f} | "
                  f"Gap={gap:+.4f}")

        except Exception as e:
            print(f"  ❌ 训练失败: {e}")
            import traceback; traceback.print_exc()

    # ── Soft-Voting物理加权融合 ────────────────────────────────
    ensemble_name = 'Soft-Voting Ensemble'
    print(f"\n[{len(models_config)+1}/{len(models_config)+1}] 终极集成 ({ensemble_name})")
    print("-" * 80)
    try:
        elite_candidates = sorted(
            [
                (n, results[n]['test_r2'], results[n]['cv_r2_mean'], results[n]['overfit_gap'])
                for n in results.keys()
                if results[n]['overfit_gap'] < 0.11   
                and results[n]['test_r2'] > 0.38       
            ],
            key=lambda x: x[1], reverse=True
        )

        print(f"  通过终极过滤的精英模型: {[n for n,_,_,_ in elite_candidates]}")

        selected_bases = elite_candidates[:3]

        if len(selected_bases) >= 2:
            model_names = [n for n,_,_,_ in selected_bases]
            print(f"  最终参战基模型: {model_names}")

            cv_scores = np.array([cv for _,_,cv,_ in selected_bases])
            cv_scores = np.maximum(cv_scores, 0.001)
            weights = cv_scores / np.sum(cv_scores)
            
            weight_dict = dict(zip(model_names, np.round(weights, 4)))
            print(f"  智能分配权重: {weight_dict}")

            ytr_st = np.zeros_like(y_train, dtype=float)
            yte_st = np.zeros_like(y_test, dtype=float)
            
            for i, name in enumerate(model_names):
                ytr_st += results[name]['y_pred_train'] * weights[i]
                yte_st += results[name]['y_pred_test']  * weights[i]

            st_tr_r2 = r2_score(y_train, ytr_st)
            st_te_r2 = r2_score(y_test,  yte_st)
            st_gap   = st_tr_r2 - st_te_r2
            st_cv_mean = np.sum([cv * w for (_,_,cv,_), w in zip(selected_bases, weights)])

            results[ensemble_name] = {
                'best_params': {'base_models': model_names, 'weights': weights.tolist()},
                'train_r2': st_tr_r2, 'test_r2': st_te_r2,
                'train_rmse': np.sqrt(mean_squared_error(y_train, ytr_st)), 'test_rmse': np.sqrt(mean_squared_error(y_test, yte_st)),
                'train_mae': mean_absolute_error(y_train, ytr_st), 'test_mae': mean_absolute_error(y_test, yte_st),
                'test_mape': np.mean(np.abs((y_test[y_test!=0] - yte_st[y_test!=0]) / y_test[y_test!=0])) * 100,
                'cv_r2_mean': st_cv_mean, 'cv_r2_std': 0.0, 'overfit_gap': st_gap, 
                'y_pred_train': ytr_st, 'y_pred_test': yte_st,
                'search_mode': 'Weighted Average (CV Based)'
            }
            
            from sklearn.dummy import DummyRegressor
            dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
            trained_models[ensemble_name] = {'model': dummy, 'use_scaled': False}
            
            print(f"融合成功！Test R²={st_te_r2:.4f} | Gap={st_gap:+.4f}")
        else:
            print("健康的基模型不足2个，跳过融合")
    except Exception as e:
        print(f"融合失败: {str(e)}")
        import traceback; traceback.print_exc()

    return results, trained_models                                   

# ==================== 5. 特征重要性分析 ====================
def analyze_feature_importance(models_dict, results, X_train, X_test, y_train, y_test, feature_cols):
    """分析特征重要性"""
    print("\n" + "=" * 100)
    print("步骤 3: 特征重要性分析")
    print("=" * 100)
    importance_results = {}
    from sklearn.inspection import permutation_importance
    
    for model_name, model_info in models_dict.items():
            
        try:
            if model_name == 'Soft-Voting Ensemble':
                base_models = results[model_name]['best_params']['base_models']
                weights = results[model_name]['best_params']['weights']
                
                # 初始化一个全零的 Series
                ensemble_imp = pd.Series(0.0, index=X_train.columns)
                valid_bases = 0
                
                for b_name, weight in zip(base_models, weights):
                    if b_name in importance_results and importance_results[b_name] is not None:
                        # 获取基模型的重要性，并归一化到 0-1 之间，再乘以权重
                        b_imp = importance_results[b_name]
                        if b_imp.sum() > 0:
                            b_imp_norm = b_imp / b_imp.sum()
                            ensemble_imp = ensemble_imp.add(b_imp_norm * weight, fill_value=0)
                            valid_bases += 1
                
                if valid_bases > 0:
                    importance_results[model_name] = ensemble_imp.sort_values(ascending=False)
                    print(f"\n{model_name} 特征重要性 (基于基模型加权):")
                    print(importance_results[model_name].head(10))
                else:
                    importance_results[model_name] = None
                continue
            # 获取训练好的模型
            model = model_info.get('trained_model', model_info['model'])
            X = X_train if model_info['use_scaled'] else X_train
            # 1. 树模型（RandomForest/XGBoost/LightGBM/GradientBoosting/ExtraTrees）
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            
            # 2. 线性模型（Ridge/Lasso/ElasticNet/Adaptive Lasso）
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                # 归一化到0-1区间
                if np.sum(importance) > 0:
                    importance = importance / np.sum(importance)
            
            # 3. KNN/SVM（排列重要性）
            else:
                print(f"  {model_name}: 计算排列重要性")
                perm_result = permutation_importance(
                    model, X, y_train, n_repeats=10, random_state=42, n_jobs=-1
                )
                importance = perm_result.importances_mean
            
            # 存储结果
            importance_results[model_name] = pd.Series(importance, index=X.columns).sort_values(ascending=False)
            
            # 打印TOP10特征
            print(f"\n{model_name} 特征重要性（TOP10）:")
            print(importance_results[model_name].head(10))
            
        except Exception as e:
            print(f"  {model_name}: 特征重要性计算失败 - {str(e)}")
            import traceback; traceback.print_exc()
            importance_results[model_name] = None
    
    # 可视化TOP5特征（以性能最优模型为例）
    best_model_name = max(models_dict.keys(), key=lambda x: models_dict[x]['test_r2'] if 'test_r2' in models_dict[x] else 0)
    if importance_results[best_model_name] is not None and importance_results[best_model_name] is not None:
        plt.figure(figsize=(12, 8))
        top5_feat = importance_results[best_model_name].head(5)
        sns.barplot(x=top5_feat.values, y=top5_feat.index, palette='viridis')
        plt.title(f'{best_model_name} 特征重要性（TOP5）', fontsize=14)
        plt.xlabel('重要性值', fontsize=12)
        plt.ylabel('特征名称', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path(f'{best_model_name}_feature_importance.png'), dpi=300)
        plt.close()
    
    return importance_results
# ==================== 6. SHAP值分析====================
def analyze_shap_values(trained_models, results, X_train, X_test, X_train_scaled,
                        X_test_scaled, feature_names, y_train, y_test):
    """
    计算所有模型的SHAP值
    """
    shap_results = {}

    print("\n" + "=" * 100)
    print("步骤 4: SHAP值分析（含集成模型物理合成）")
    print("=" * 100)

    # 优先计算基础模型
    model_names_sorted = [m for m in trained_models.keys() if m != 'Soft-Voting Ensemble']
    if 'Soft-Voting Ensemble' in trained_models:
        model_names_sorted.append('Soft-Voting Ensemble')

    for model_name in model_names_sorted:
        model_info = trained_models[model_name]
        print(f"\n计算 {model_name} 的SHAP值...")
        print("-" * 80)

        model     = model_info['model']
        use_scaled= model_info['use_scaled']
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled  if use_scaled else X_test

        sv          = None   # shap values (2D array)
        base_val    = None
        X_sample    = None
        explainer   = None

        # =====================================================================
        # 1. Soft-Voting Ensemble 的 SHAP 物理加权计算
        # =====================================================================
        if model_name == 'Soft-Voting Ensemble':
            try:
                base_models = results[model_name]['best_params']['base_models']
                weights = results[model_name]['best_params']['weights']
                
                # 检查基模型的 SHAP 是否都存在
                missing_shap = [m for m in base_models if m not in shap_results]
                if missing_shap:
                    print(f"  ✗ 缺少基模型 {missing_shap} 的 SHAP 值，无法合成 Ensemble SHAP")
                    continue
                
                # 以第一个基模型的样本作为参考
                ref_model = base_models[0]
                sv_shape = shap_results[ref_model]['shap_values'].shape
                sv = np.zeros(sv_shape)
                base_val = 0.0
                X_sample = shap_results[ref_model]['X_sample']
                
                # 遍历基模型，严格按权重加权求和
                for b_name, weight in zip(base_models, weights):
                    b_sv = shap_results[b_name]['shap_values']
                    b_base = shap_results[b_name]['base_value']
                    sv += b_sv * weight
                    base_val += b_base * weight
                
                explainer = None  # 集成模型没有单一的 Explainer 对象
                print(f"  ✓ Soft-Voting 物理加权 SHAP 合成成功！(基于 {base_models})")
                
            except Exception as e:
                print(f"  ✗ Soft-Voting SHAP 合成失败: {e}")
                continue

        # =====================================================================
        # 2. 树模型 — TreeExplainer
        # =====================================================================
        elif model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting',
                          'Extra Trees', 'LightGBM']:
            try:
                explainer = shap.TreeExplainer(model)
                X_sample  = X_te   # 全量测试集
                raw       = explainer.shap_values(X_sample)

                sv = raw[0] if isinstance(raw, list) else raw
                sv = np.array(sv)

                ev = explainer.expected_value
                base_val = float(ev[0] if hasattr(ev, '__len__') else ev)

                print(f"  ✓ TreeExplainer 成功 (样本数: {len(X_sample)})")

            except Exception as e:
                print(f"  ✗ TreeExplainer 失败: {e}")
                try:
                    print("  → 降级到 PermutationExplainer...")
                    bg = shap.sample(X_tr, min(50, len(X_tr)))
                    explainer = shap.PermutationExplainer(model.predict, bg)
                    X_sample  = X_te.iloc[:min(80, len(X_te))]
                    sv_obj    = explainer(X_sample)
                    sv        = np.array(sv_obj.values)
                    base_val  = float(sv_obj.base_values[0] if hasattr(sv_obj.base_values, '__len__') else sv_obj.base_values)
                    print(f"  ✓ PermutationExplainer 成功 (样本数: {len(X_sample)})")
                except Exception as e2:
                    print(f"  ✗ 降级也失败: {e2}")
                    continue

        # =====================================================================
        # 3. 线性模型 — LinearExplainer
        # =====================================================================
        elif model_name in ['Ridge', 'Lasso', 'Adaptive Lasso', 'ElasticNet']:
            try:
                explainer = shap.LinearExplainer(
                    model, X_tr,
                    feature_perturbation='interventional'
                )
                X_sample = X_te
                raw      = explainer.shap_values(X_sample)
                sv       = np.array(raw[0] if isinstance(raw, list) else raw)

                ev = explainer.expected_value
                base_val = float(ev[0] if hasattr(ev, '__len__') else ev)

                print(f"  ✓ LinearExplainer 成功 (样本数: {len(X_sample)})")

            except Exception as e:
                print(f"  ✗ LinearExplainer 失败: {e}")
                continue

        # =====================================================================
        # 4. SVM / KNN — KernelExplainer
        # =====================================================================
        elif model_name in ['SVM', 'KNN']:
            try:
                n_bg = min(20, len(X_tr))
                background = shap.kmeans(X_tr, n_bg)
                explainer  = shap.KernelExplainer(model.predict, background)

                n_explain = min(60, len(X_te))
                X_sample  = X_te.iloc[:n_explain]
                raw       = explainer.shap_values(X_sample, silent=True)
                sv        = np.array(raw[0] if isinstance(raw, list) else raw)

                ev = explainer.expected_value
                base_val = float(ev[0] if hasattr(ev, '__len__') else ev)

                print(f"  ✓ KernelExplainer (kmeans背景={n_bg}) 成功 (样本数: {n_explain})")

            except Exception as e:
                print(f"  ✗ KernelExplainer 失败: {e}")
                continue

        else:
            try:
                bg   = shap.sample(X_tr, min(20, len(X_tr)))
                explainer = shap.PermutationExplainer(model.predict, bg)
                n_explain = min(60, len(X_te))
                X_sample  = X_te.iloc[:n_explain]
                sv_obj    = explainer(X_sample)
                sv        = np.array(sv_obj.values)
                base_val  = float(
                    sv_obj.base_values[0] if hasattr(sv_obj.base_values, '__len__') else sv_obj.base_values
                )
                print(f"  ✓ PermutationExplainer (通用) 成功")
            except Exception as e:
                print(f"  ✗ 通用SHAP失败: {e}")
                continue

        # =====================================================================
        # 后处理：计算统计量并强制存入
        # =====================================================================
        if sv is None or sv.ndim != 2:
            print(f"  ✗ SHAP值维度异常或为空，跳过 {model_name}")
            continue

        if hasattr(X_sample, 'columns'):
            feat_names_used = list(X_sample.columns)
        else:
            feat_names_used = feature_names[:sv.shape[1]]

        if sv.shape[1] != len(feat_names_used):
            n_min = min(sv.shape[1], len(feat_names_used))
            sv = sv[:, :n_min]
            feat_names_used = feat_names_used[:n_min]

        mean_abs_shap   = np.abs(sv).mean(axis=0)
        mean_signed_shap= sv.mean(axis=0)

        shap_importance = pd.DataFrame({
            'Feature':           feat_names_used,
            'Mean_Abs_SHAP':     mean_abs_shap,
            'Mean_Signed_SHAP':  mean_signed_shap,
        }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

        shap_importance['Direction'] = shap_importance['Mean_Signed_SHAP'].apply(
            lambda x: '正影响(↑)' if x >= 0 else '负影响(↓)'
        )

        # 存入字典
        shap_results[model_name] = {
            'explainer':          explainer,
            'shap_values':        sv,
            'X_sample':           X_sample,
            'base_value':         base_val,
            'importance':         shap_importance,
            'feature_names_used': feat_names_used
        }

        n_samp = len(X_sample) if hasattr(X_sample, '__len__') else '?'
        print(f"✓ SHAP计算完成 | 样本数: {n_samp} | 特征数: {sv.shape[1]}")
        print("  Top 10 特征 (含影响方向):")
        for _, row in shap_importance.head(10).iterrows():
            bar = '█' * int(row['Mean_Abs_SHAP'] / (mean_abs_shap.max()+1e-9) * 20)
            print(f"    {row['Feature']:<30} {row['Mean_Abs_SHAP']:.5f} {bar} {row['Direction']}")

    success = len(shap_results)
    total   = len(trained_models)
    print(f"\nSHAP分析完成: {success}/{total} 个模型成功")
    return shap_results
# ==================== 7. 可视化函数 ====================
def create_visualizations(results, importance_results, shap_results, 
                         feature_names, y_test):
    """创建所有可视化图表"""
    
    print("\n" + "=" * 100)
    print("步骤 5: 生成可视化图表")
    print("=" * 100)
    
    # 7.1 模型性能对比
    print("\n生成模型性能对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    model_names = list(results.keys())
    train_r2 = [results[m]['train_r2'] for m in model_names]
    test_r2 = [results[m]['test_r2'] for m in model_names]
    test_rmse = [results[m]['test_rmse'] for m in model_names]
    test_mae = [results[m]['test_mae'] for m in model_names]
    test_mape = [results[m]['test_mape'] for m in model_names]
    cv_means = [results[m]['cv_r2_mean'] for m in model_names]
    cv_stds = [results[m]['cv_r2_std'] for m in model_names]
    
    # R² 对比
    x = np.arange(len(model_names))
    width = 0.35
    axes[0, 0].bar(x - width/2, train_r2, width, label='训练集', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, test_r2, width, label='测试集', alpha=0.8, color='coral')
    axes[0, 0].set_xlabel('模型', fontsize=11)
    axes[0, 0].set_ylabel('R² 分数', fontsize=11)
    axes[0, 0].set_title('模型R²性能对比', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='优秀线')
    
    # RMSE 对比
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = axes[0, 1].bar(model_names, test_rmse, alpha=0.8, color=colors)
    axes[0, 1].set_xlabel('模型', fontsize=11)
    axes[0, 1].set_ylabel('RMSE', fontsize=11)
    axes[0, 1].set_title('模型RMSE对比 (测试集)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # MAE 对比
    bars = axes[0, 2].bar(model_names, test_mae, alpha=0.8, color='lightgreen')
    axes[0, 2].set_xlabel('模型', fontsize=11)
    axes[0, 2].set_ylabel('MAE', fontsize=11)
    axes[0, 2].set_title('模型MAE对比 (测试集)', fontsize=13, fontweight='bold')
    axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[0, 2].grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 交叉验证R²
    bars = axes[1, 0].bar(model_names, cv_means, yerr=cv_stds, alpha=0.8, 
                          color='skyblue', capsize=5, error_kw={'linewidth': 2})
    axes[1, 0].set_xlabel('模型', fontsize=11)
    axes[1, 0].set_ylabel('交叉验证 R²', fontsize=11)
    axes[1, 0].set_title('10折交叉验证R²对比', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # MAPE 对比
    bars = axes[1, 1].bar(model_names, test_mape, alpha=0.8, color='salmon')
    axes[1, 1].set_xlabel('模型', fontsize=11)
    axes[1, 1].set_ylabel('MAPE (%)', fontsize=11)
    axes[1, 1].set_title('模型MAPE对比 (测试集)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 过拟合分析
    overfit_gaps = [train_r2[i] - test_r2[i] for i in range(len(model_names))]
    colors_overfit = ['red' if gap > 0.15 else 'orange' if gap > 0.10 else 'green' 
                      for gap in overfit_gaps]
    bars = axes[1, 2].bar(model_names, overfit_gaps, alpha=0.8, color=colors_overfit)
    axes[1, 2].set_xlabel('模型', fontsize=11)
    axes[1, 2].set_ylabel('训练集R² - 测试集R²', fontsize=11)
    axes[1, 2].set_title('过拟合分析', fontsize=13, fontweight='bold')
    axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[1, 2].axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='轻微过拟合线')
    axes[1, 2].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='严重过拟合线')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path('model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: model_performance_comparison.png")
    plt.close()
    
    # 7.2 预测值 vs 真实值
    print("生成预测值vs真实值散点图...")
    n_models = len(model_names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig_height = max(6, min(6*n_rows, 30))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_height))
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        y_pred = results[model_name]['y_pred_test']
        r2 = results[model_name]['test_r2']
        rmse = results[model_name]['test_rmse']
        mae = results[model_name]['test_mae']
        
        # 散点图
        axes[idx].scatter(y_test, y_pred, alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # 理想预测线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 
                       'r--', lw=2, label='理想预测线', alpha=0.8)
        
        # 添加拟合线
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        axes[idx].plot(y_test.sort_values(), p(y_test.sort_values()), 
                      "g-", alpha=0.5, linewidth=2, label='拟合线')
        
        axes[idx].set_xlabel('真实值', fontsize=11)
        axes[idx].set_ylabel('预测值', fontsize=11)
        axes[idx].set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}', 
                           fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(alpha=0.3)
        
        # 添加统计信息
        textstr = f'样本数: {len(y_test)}\n相关系数: {np.corrcoef(y_test, y_pred)[0,1]:.4f}'
        axes[idx].text(0.05, 0.95, textstr, transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path('prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: prediction_vs_actual.png")
    plt.close()
    
    # 7.3 残差分析图
    print("生成残差分析图...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_height))
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        y_pred = results[model_name]['y_pred_test']
        residuals = y_test - y_pred
        
        # 残差散点图
        axes[idx].scatter(y_pred, residuals, alpha=0.6, s=30, c='purple', edgecolors='black', linewidth=0.5)
        axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[idx].set_xlabel('预测值', fontsize=11)
        axes[idx].set_ylabel('残差', fontsize=11)
        axes[idx].set_title(f'{model_name} - 残差分析', fontsize=11, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        
        # 添加残差统计信息
        textstr = f'残差均值: {residuals.mean():.4f}\n残差标准差: {residuals.std():.4f}'
        axes[idx].text(0.05, 0.95, textstr, transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path('residual_analysis.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: residual_analysis.png")
    plt.close()
    
    # 7.4 特征重要性可视化
    if importance_results:
        print("生成特征重要性图...")
        n_importance = len(importance_results)
        n_cols = 3
        n_rows = (n_importance + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        if n_importance == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        for idx, (model_name, importance_df) in enumerate(importance_results.items()):
            if importance_df is not None:
                top_features = importance_df.head(15)
                
                # 水平条形图
                y_pos = np.arange(len(top_features))
                colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
                
                axes[idx].barh(y_pos, top_features.values, color=colors_bar, alpha=0.8)
                axes[idx].set_yticks(y_pos)
                axes[idx].set_yticklabels(top_features.index, fontsize=9)
                axes[idx].set_xlabel('重要性', fontsize=11)
                axes[idx].set_title(f'{model_name} - Top 15 特征', fontsize=12, fontweight='bold')
                axes[idx].invert_yaxis()
                axes[idx].grid(axis='x', alpha=0.3)
                
                # 添加数值标签
                for i, v in enumerate(top_features.values):
                    axes[idx].text(v, i, f' {v:.4f}', va='center', fontsize=8)
        
        # 隐藏多余的子图
        for idx in range(len(importance_results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(out_path('feature_importance.png'), dpi=300, bbox_inches='tight')
        print("✓ 保存: feature_importance.png")
        plt.close()
    
    # 7.5 特征重要性热力图
    if importance_results and len(importance_results) > 1:
        print("生成特征重要性热力图...")
        
        # 创建特征重要性矩阵
        all_features = set()
        for imp_df in importance_results.values():
            if imp_df is not None:
                all_features.update(imp_df.index.tolist())
        
        importance_matrix = pd.DataFrame(index=sorted(all_features), 
                                        columns=importance_results.keys())
        
        for model_name, imp_df in importance_results.items():
            if imp_df is not None:
                for feature, importance in imp_df.items():
                    importance_matrix.loc[feature, model_name] = importance
        
        importance_matrix = importance_matrix.fillna(0)
        
        # 标准化每列
        importance_matrix_norm = importance_matrix.div(importance_matrix.max(axis=0), axis=1)
        
        # 选择Top特征
        top_features_overall = importance_matrix_norm.sum(axis=1).nlargest(20).index
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(importance_matrix_norm.loc[top_features_overall], 
                   annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': '标准化重要性'},
                   linewidths=0.5)
        plt.title('特征重要性热力图 (Top 20特征, 跨模型对比)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_path('feature_importance_heatmap.png'), dpi=300, bbox_inches='tight')
        print("✓ 保存: feature_importance_heatmap.png")
        plt.close()
    
    # 7.6 SHAP可视化
    if shap_results:
        print("生成SHAP可视化图...")
        
        # 找出最优的树模型
        tree_models_in_shap = [m for m in shap_results 
                               if m in ['Random Forest','XGBoost','Gradient Boosting',
                                        'Extra Trees','LightGBM']]
        best_tree_for_shap = None
        if tree_models_in_shap and results:
            best_tree_for_shap = max(
                tree_models_in_shap,
                key=lambda m: results.get(m, {}).get('test_r2', -999)
            )
        
        for model_name, shap_data in shap_results.items():
            try:
                sv       = shap_data['shap_values']
                X_samp   = shap_data['X_sample']
                feat_names_used = shap_data.get('feature_names_used', feature_names)
                base_val = shap_data['base_value']
                
                # --- (a) SHAP Summary Plot（蜂群图）---
                plt.figure(figsize=(12, 8))
                shap.summary_plot(sv, X_samp, feature_names=feat_names_used,
                                  show=False, max_display=18)
                plt.title(f'{model_name} — SHAP Summary Plot (蜂群图)',
                         fontsize=13, fontweight='bold', pad=15)
                plt.tight_layout()
                plt.savefig(out_path(f'shap_summary_{model_name.replace(" ", "_")}.png'),
                           dpi=300, bbox_inches='tight')
                print(f"✓ shap_summary_{model_name.replace(' ', '_')}.png")
                plt.close()
                
                # --- (b) SHAP Bar Plot（带正负方向颜色）---
                imp_df = shap_data['importance']
                top_n = min(18, len(imp_df))
                top_imp = imp_df.head(top_n)
                
                fig, ax = plt.subplots(figsize=(11, 8))
                bar_colors = ['#e74c3c' if d == '正影响(↑)' else '#3498db'
                              for d in top_imp['Direction']]
                y_pos = np.arange(top_n)
                bars = ax.barh(y_pos, top_imp['Mean_Abs_SHAP'], color=bar_colors, alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_imp['Feature'], fontsize=10)
                ax.invert_yaxis()
                ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
                ax.set_title(f'{model_name} — SHAP Feature Importance (红=正影响, 蓝=负影响)',
                            fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                for bar, val in zip(bars, top_imp['Mean_Abs_SHAP']):
                    ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y()+bar.get_height()/2,
                           f'{val:.4f}', va='center', fontsize=8)
                # 图例
                from matplotlib.patches import Patch
                legend_elems = [Patch(facecolor='#e74c3c', label='正影响(增大PM)'),
                                Patch(facecolor='#3498db', label='负影响(减小PM)')]
                ax.legend(handles=legend_elems, loc='lower right', fontsize=10)
                plt.tight_layout()
                plt.savefig(out_path(f'shap_bar_{model_name.replace(" ", "_")}.png'),
                           dpi=300, bbox_inches='tight')
                print(f"✓ shap_bar_{model_name.replace(' ', '_')}.png")
                plt.close()
                
                # --- (c) SHAP Waterfall Plot（最高预测值样本）---
                try:
                    n_sv = sv.shape[0]
                    approx_pred = sv.sum(axis=1) + base_val
                    highest_idx = int(np.argmax(approx_pred))
                    
                    if hasattr(X_samp, 'iloc'):
                        sample_data = X_samp.iloc[highest_idx].values
                    else:
                        sample_data = X_samp[highest_idx]
                    
                    plt.figure(figsize=(11, 8))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=sv[highest_idx],
                            base_values=base_val,
                            data=sample_data,
                            feature_names=feat_names_used
                        ),
                        show=False, max_display=15
                    )
                    plt.title(f'{model_name} — SHAP Waterfall (预测值最高样本)',
                             fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(out_path(f'shap_waterfall_{model_name.replace(" ", "_")}.png'),
                               dpi=300, bbox_inches='tight')
                    print(f"✓ shap_waterfall_{model_name.replace(' ', '_')}.png")
                    plt.close()
                except Exception as ew:
                    print(f"  Waterfall图跳过: {ew}")
                
                                # --- (d) SHAP 依赖图（Top 4特征，仅最优树模型）---
                if model_name == best_tree_for_shap:
                    top4_features = imp_df['Feature'].head(4).tolist()
                    fig, axes_dep = plt.subplots(2, 2, figsize=(16, 12))
                    axes_dep = axes_dep.flatten()
                    
                    for fi, feat in enumerate(top4_features):
                        if feat in feat_names_used:
                            feat_idx = feat_names_used.index(feat)
                            ax_dep = axes_dep[fi]
                            
                            # 获取该特征的特征值和SHAP值
                            feat_vals = X_samp[feat].values if hasattr(X_samp,'columns') else X_samp[:, feat_idx]
                            shap_vals_feat = sv[:, feat_idx]
                            
                            # 用Ground_PM作为交互色彩变量（若存在）
                            color_feat = 'Ground_PM' if 'Ground_PM' in feat_names_used else None
                            if color_feat and color_feat != feat:
                                color_idx = feat_names_used.index(color_feat)
                                c_vals = X_samp[color_feat].values if hasattr(X_samp,'columns') else X_samp[:, color_idx]
                                sc = ax_dep.scatter(feat_vals, shap_vals_feat,
                                                   c=c_vals, cmap='RdYlBu_r', alpha=0.7, s=20)
                                plt.colorbar(sc, ax=ax_dep, label=color_feat)
                            else:
                                ax_dep.scatter(feat_vals, shap_vals_feat,
                                             alpha=0.6, s=20, c='steelblue')
                            
                            ax_dep.axhline(0, color='gray', linestyle='--', lw=1)
                            ax_dep.set_xlabel(feat, fontsize=11)
                            ax_dep.set_ylabel('SHAP value', fontsize=11)
                            ax_dep.set_title(f'依赖图: {feat}', fontsize=12, fontweight='bold')
                            ax_dep.grid(alpha=0.3)
                    
                    plt.suptitle(f'{model_name} — SHAP依赖图 (Top 4重要特征)',
                                fontsize=14, fontweight='bold', y=1.01)
                    plt.tight_layout()
                    plt.savefig(out_path(f'shap_dependence_{model_name.replace(" ", "_")}.png'),
                               dpi=300, bbox_inches='tight')
                    print(f"✓ shap_dependence_{model_name.replace(' ', '_')}.png")
                    plt.close()
                    
            except Exception as e:
                print(f"⚠️  {model_name} SHAP可视化失败: {str(e)}")
                continue
        
        # --- (e) 双最优模型 SHAP 对比图 ---
        if len(shap_results) >= 2 and results:
            print("生成双模型SHAP对比图...")
            sorted_by_r2 = sorted(
                [m for m in shap_results if m in results],
                key=lambda m: results[m]['test_r2'], reverse=True
            )
            top2 = sorted_by_r2[:2]
            
            try:
                fig, axes2 = plt.subplots(1, 2, figsize=(22, 9))
                for ai, mn in enumerate(top2):
                    shap_importance_matrix = pd.DataFrame()
                    for mn2, sd2 in shap_results.items():
                        shap_importance_matrix[mn2] = (
                            sd2['importance'].set_index('Feature')['Mean_Abs_SHAP']
                        )
                    
                    top_feats_all = (shap_importance_matrix
                                    .fillna(0).sum(axis=1).nlargest(15).index.tolist())
                    
                    sd   = shap_results[mn]
                    imp  = sd['importance'].set_index('Feature')['Mean_Abs_SHAP']
                    vals = [imp.get(f, 0) for f in top_feats_all]
                    
                    # 重新查方向
                    dir_map = dict(zip(sd['importance']['Feature'], sd['importance']['Direction']))
                    colors_bar = ['#e74c3c' if dir_map.get(f,'正影响(↑)')=='正影响(↑)' else '#3498db'
                                  for f in top_feats_all]
                    
                    y_p = np.arange(len(top_feats_all))
                    axes2[ai].barh(y_p, vals, color=colors_bar, alpha=0.85)
                    axes2[ai].set_yticks(y_p)
                    axes2[ai].set_yticklabels(top_feats_all, fontsize=9)
                    axes2[ai].invert_yaxis()
                    axes2[ai].set_xlabel('mean(|SHAP value|)', fontsize=11)
                    r2v = results[mn]['test_r2']
                    axes2[ai].set_title(f'{mn} (Test R²={r2v:.4f})', fontsize=12, fontweight='bold')
                    axes2[ai].grid(axis='x', alpha=0.3)
                
                plt.suptitle('Top-2模型 SHAP特征重要性对比 (红=正影响, 蓝=负影响)',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(out_path('shap_top2_comparison.png'), dpi=300, bbox_inches='tight')
                print("✓ shap_top2_comparison.png")
                plt.close()
            except Exception as ec:
                print(f"  双模型对比图跳过: {ec}")
        
        # --- (f) SHAP重要性热力图---
        print("生成SHAP重要性热力图...")
        shap_importance_matrix = pd.DataFrame()
        for mn2, sd2 in shap_results.items():
            shap_importance_matrix[mn2] = (
                sd2['importance'].set_index('Feature')['Mean_Abs_SHAP']
            )
        
        if not shap_importance_matrix.empty:
            shap_importance_norm = shap_importance_matrix.fillna(0)
            shap_importance_norm = shap_importance_norm.div(
                shap_importance_norm.max(axis=0).replace(0, 1), axis=1
            )
            top_shap_features = shap_importance_norm.sum(axis=1).nlargest(20).index
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(shap_importance_norm.loc[top_shap_features],
                       annot=True, fmt='.3f', cmap='Blues',
                       cbar_kws={'label': '标准化SHAP重要性'},
                       linewidths=0.5)
            plt.title('SHAP重要性热力图 (Top 20特征, 跨模型对比)',
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('模型', fontsize=12)
            plt.ylabel('特征', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(out_path('shap_importance_heatmap.png'), dpi=300, bbox_inches='tight')
            print("✓ shap_importance_heatmap.png")
            plt.close()
    
    # 7.7 模型排名雷达图
    print("生成模型性能雷达图...")
    
    # 准备数据（标准化到0-1）
    metrics = {
        'R²': test_r2,
        'RMSE': [1/(1+x) for x in test_rmse],  
        'MAE': [1/(1+x) for x in test_mae],    
        'CV R²': cv_means,
        '泛化能力': [1-abs(train_r2[i]-test_r2[i]) for i in range(len(model_names))]
    }
    
    # 标准化
    for key in metrics:
        max_val = max(metrics[key])
        min_val = min(metrics[key])
        if max_val > min_val:
            metrics[key] = [(x-min_val)/(max_val-min_val) for x in metrics[key]]
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    colors_radar = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for idx, model_name in enumerate(model_names):
        values = [metrics[key][idx] for key in metrics.keys()]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics.keys(), fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('模型综合性能雷达图', fontsize=14, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig(out_path('model_radar_chart.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: model_radar_chart.png")
    plt.close()

    # 7.8 过拟合诊断专项图
    print("生成过拟合诊断图...")
    gaps = [results[m]['train_r2'] - results[m]['test_r2'] for m in model_names]
    cv_r2 = [results[m]['cv_r2_mean'] for m in model_names]

    fig, axes_of = plt.subplots(1, 3, figsize=(22, 7))

    # (1) 训练/测试/CV R² 三线对比
    x = np.arange(len(model_names))
    w = 0.25
    axes_of[0].bar(x - w, train_r2, w, label='训练集R²',  color='#3498db', alpha=0.85)
    axes_of[0].bar(x,     test_r2,  w, label='测试集R²',  color='#e74c3c', alpha=0.85)
    axes_of[0].bar(x + w, cv_r2,    w, label='CV R²(均值)', color='#2ecc71', alpha=0.85)
    axes_of[0].set_xticks(x)
    axes_of[0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes_of[0].set_ylabel('R²', fontsize=11)
    axes_of[0].set_title('训练/测试/CV R² 三向对比', fontsize=12, fontweight='bold')
    axes_of[0].legend(fontsize=9)
    axes_of[0].axhline(0.8, color='gray', ls='--', lw=1, alpha=0.5)
    axes_of[0].grid(axis='y', alpha=0.3)

    # (2) 过拟合差距条形图（颜色编码）
    gap_colors = []
    for g in gaps:
        if   g > 0.20: gap_colors.append('#c0392b')   # 深红：严重
        elif g > 0.12: gap_colors.append('#e67e22')   # 橙：明显
        elif g > 0.05: gap_colors.append('#f1c40f')   # 黄：轻微
        else:          gap_colors.append('#27ae60')   # 绿：良好
    bars_gap = axes_of[1].bar(model_names, gaps, color=gap_colors, alpha=0.85, edgecolor='white')
    axes_of[1].axhline(0.20, color='#c0392b', ls='--', lw=1.5, label='严重阈值(0.20)')
    axes_of[1].axhline(0.12, color='#e67e22', ls='--', lw=1.5, label='明显阈值(0.12)')
    axes_of[1].axhline(0.05, color='#f1c40f', ls='--', lw=1.5, label='轻微阈值(0.05)')
    for bar, g in zip(bars_gap, gaps):
        axes_of[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{g:+.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes_of[1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes_of[1].set_ylabel('过拟合差距（训练R² - 测试R²）', fontsize=11)
    axes_of[1].set_title('各模型过拟合差距（颜色=严重程度）', fontsize=12, fontweight='bold')
    axes_of[1].grid(axis='y', alpha=0.3)
    # 图例补丁
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color='#27ae60', label='🟢 泛化良好 (≤0.05)'),
        Patch(color='#f1c40f', label='🟡 轻微过拟合 (0.05-0.12)'),
        Patch(color='#e67e22', label='🟠 明显过拟合 (0.12-0.20)'),
        Patch(color='#c0392b', label='🔴 严重过拟合 (>0.20)')
    ]
    axes_of[1].legend(handles=legend_patches, fontsize=8, loc='upper left')

    # (3) 测试R² vs CV R² 散点图（理想=对角线）
    axes_of[2].scatter(cv_r2, test_r2, c=gap_colors, s=120, edgecolors='black', linewidth=0.8, zorder=5)
    min_v = min(min(cv_r2), min(test_r2)) - 0.05
    max_v = max(max(cv_r2), max(test_r2)) + 0.05
    axes_of[2].plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, alpha=0.5, label='理想线(CV=Test)')
    for mn2, cx, ty in zip(model_names, cv_r2, test_r2):
        axes_of[2].annotate(mn2, (cx, ty), textcoords='offset points',
                            xytext=(4, 4), fontsize=7)
    axes_of[2].set_xlabel('CV R²（交叉验证均值）', fontsize=11)
    axes_of[2].set_ylabel('测试集R²', fontsize=11)
    axes_of[2].set_title('CV R² vs 测试集R²（偏离对角线=泛化差）', fontsize=12, fontweight='bold')
    axes_of[2].legend(fontsize=9)
    axes_of[2].grid(alpha=0.3)

    plt.suptitle('模型过拟合诊断综合分析', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path('overfitting_diagnosis.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: overfitting_diagnosis.png")
    plt.close()

    # 7.9 误差分布箱线图
    print("生成误差分布箱线图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绝对误差
    abs_errors = [np.abs(y_test - results[m]['y_pred_test']) for m in model_names]
    bp1 = axes[0].boxplot(abs_errors, labels=model_names, patch_artist=True,
                          showmeans=True, meanline=True)
    for patch, color in zip(bp1['boxes'], colors_radar):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_xlabel('模型', fontsize=12)
    axes[0].set_ylabel('绝对误差', fontsize=12)
    axes[0].set_title('模型绝对误差分布', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 相对误差百分比
    rel_errors = [np.abs((y_test - results[m]['y_pred_test']) / y_test) * 100 
                  for m in model_names]
    bp2 = axes[1].boxplot(rel_errors, labels=model_names, patch_artist=True,
                          showmeans=True, meanline=True)
    for patch, color in zip(bp2['boxes'], colors_radar):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_xlabel('模型', fontsize=12)
    axes[1].set_ylabel('相对误差 (%)', fontsize=12)
    axes[1].set_title('模型相对误差分布', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(out_path('error_distribution.png'), dpi=300, bbox_inches='tight')
    print("✓ 保存: error_distribution.png")
    plt.close()
    
    print("\n所有可视化图表生成完成！")

# ==================== 8. 生成分析报告 ====================
def generate_report(results, importance_results, shap_results, feature_names):
    """生成详细的分析报告"""
    report = []
    report.append("=" * 120)
    report.append("地铁颗粒物浓度预测分析报告")
    report.append("=" * 120)
    report.append("")
  
    # 一、数据概况
    report.append("一、数据概况")
    report.append("-" * 120)
    report.append(f"特征数量: {len(feature_names)}")
    report.append(f"模型数量: {len(results)}")
    report.append("")
  
    # 二、模型性能对比
    report.append("二、模型性能对比")
    report.append("-" * 120)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    report.append(f"{'排名':<6} {'模型':<22} {'训练R²':<10} {'测试R²':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'CV R²':<12} {'过拟合Gap':<12}")
    report.append("-" * 120)
  
    for rank, (model_name, model_results) in enumerate(sorted_models, 1):
        gap = model_results['overfit_gap']
        gap_status = "良好" if gap <= 0.05 else "轻微" if gap <= 0.12 else "明显" if gap <= 0.20 else "严重"
        report.append(
            f"{rank:<6} {model_name:<22} {model_results['train_r2']:<10.4f} "
            f"{model_results['test_r2']:<10.4f} {model_results['test_rmse']:<10.4f} "
            f"{model_results['test_mae']:<10.4f} {model_results['test_mape']:<10.2f}% "
            f"{model_results['cv_r2_mean']:<12.4f} {gap:+.4f}({gap_status})"
        )
    report.append("")
  
    # 三、最佳模型详情
    best_model_name = sorted_models[0][0]
    best_results = sorted_models[0][1]
    report.append("三、最佳模型详情")
    report.append("-" * 120)
    report.append(f"模型名称: {best_model_name}")
    report.append(f"测试集R²: {best_results['test_r2']:.4f}")
    report.append(f"测试集RMSE: {best_results['test_rmse']:.4f}")
    report.append(f"测试集MAE: {best_results['test_mae']:.4f}")
    report.append(f"测试集MAPE: {best_results['test_mape']:.2f}%")
    report.append(f"交叉验证R²: {best_results['cv_r2_mean']:.4f} ± {best_results['cv_r2_std']:.4f}")
    report.append(f"过拟合差距: {best_results['overfit_gap']:+.4f}")
    report.append("")
    report.append("最佳超参数:")
    for param, value in best_results['best_params'].items():
        report.append(f"  {param}: {value}")
    report.append("")
  
    # 四、特征重要性分析
    if importance_results and best_model_name in importance_results:
        report.append("四、特征重要性分析（基于最佳模型）")
        report.append("-" * 120)
        imp = importance_results[best_model_name]
        if imp is not None:
            report.append(f"{'排名':<6} {'特征名称':<35} {'重要性值':<15}")
            report.append("-" * 60)
            for rank, (feat, val) in enumerate(imp.head(15).items(), 1):
                report.append(f"{rank:<6} {feat:<35} {val:<15.6f}")
        report.append("")
  
    # 五、SHAP值分析
    if shap_results and best_model_name in shap_results:
        report.append("五、SHAP值分析（基于最佳模型）")
        report.append("-" * 120)
        shap_imp = shap_results[best_model_name].get('importance')
        if shap_imp is not None:
            report.append(f"{'排名':<6} {'特征名称':<35} {'Mean |SHAP|':<15} {'影响方向':<12}")
            report.append("-" * 70)
            for rank, (_, row) in enumerate(shap_imp.head(15).iterrows(), 1):
                report.append(
                    f"{rank:<6} {row['Feature']:<35} {row['Mean_Abs_SHAP']:<15.6f} {row['Direction']:<12}"
                )
        report.append("")
  
    # 六、模型泛化能力评估
    report.append("六、模型泛化能力评估")
    report.append("-" * 120)
    good_models = [m for m, r in results.items() if r['overfit_gap'] <= 0.05]
    mild_models = [m for m, r in results.items() if 0.05 < r['overfit_gap'] <= 0.12]
    severe_models = [m for m, r in results.items() if r['overfit_gap'] > 0.12]
  
    report.append(f"泛化良好模型 (Gap≤0.05): {len(good_models)} 个")
    for m in good_models:
        report.append(f"  • {m}: Gap={results[m]['overfit_gap']:+.4f}")
    report.append("")
    
    report.append(f"轻微过拟合模型 (0.05<Gap≤0.12): {len(mild_models)} 个")
    for m in mild_models:
        report.append(f"  • {m}: Gap={results[m]['overfit_gap']:+.4f}")
    report.append("")
    
    report.append(f"明显过拟合模型 (Gap>0.12): {len(severe_models)} 个")
    for m in severe_models:
        report.append(f"  • {m}: Gap={results[m]['overfit_gap']:+.4f}")
    report.append("")
    
    # 七、交叉验证稳定性分析
    report.append("七、交叉验证稳定性分析")
    report.append("-" * 120)
    report.append(f"{'模型':<22} {'CV R²均值':<12} {'CV R²标准差':<15} {'稳定性评价':<15}")
    report.append("-" * 70)
    for model_name, model_results in sorted_models:
        cv_std = model_results['cv_r2_std']
        stability = "优秀" if cv_std < 0.02 else "良好" if cv_std < 0.05 else "一般" if cv_std < 0.10 else "较差"
        report.append(
            f"{model_name:<22} {model_results['cv_r2_mean']:<12.4f} "
            f"{cv_std:<15.4f} {stability:<15}"
        )
    report.append("")
    
    # 八、关键发现与建议
    report.append("八、关键发现与建议")
    report.append("-" * 120)
    report.append("")
    report.append("【关键发现】")
    report.append(f"1. 最佳模型为 {best_model_name}，测试集R²达到 {best_results['test_r2']:.4f}")
    report.append(f"2. 模型预测误差RMSE为 {best_results['test_rmse']:.4f}，MAE为 {best_results['test_mae']:.4f}")
    report.append(f"3. 交叉验证R²为 {best_results['cv_r2_mean']:.4f}，显示出良好的稳定性")
    
    if importance_results and best_model_name in importance_results:
        imp = importance_results[best_model_name]
        if imp is not None and len(imp) > 0:
            top_feat = imp.index[0]
            report.append(f"4. 最重要的特征是 {top_feat}，对预测结果影响最大")
    
    report.append("")
    report.append("【改进建议】")
    if best_results['overfit_gap'] > 0.10:
        report.append("1. 模型存在一定过拟合，建议:")
        report.append("   • 增加正则化强度")
        report.append("   • 减少模型复杂度（如降低树深度）")
        report.append("   • 收集更多训练数据")
    else:
        report.append("1. 模型泛化能力良好，可考虑:")
        report.append("   • 适当增加模型复杂度以提升性能")
        report.append("   • 尝试更多特征工程")
    
    report.append("")
    report.append("2. 特征工程建议:")
    report.append("   • 基于SHAP分析结果，重点关注高贡献特征")
    report.append("   • 考虑添加更多交互特征")
    report.append("   • 对低重要性特征进行筛选")
    
    report.append("")
    report.append("3. 模型部署建议:")
    report.append(f"   • 推荐使用 {best_model_name} 进行生产部署")
    report.append("   • 建立模型监控机制，定期评估预测性能")
    report.append("   • 根据新数据定期重训练模型")
    
    report.append("")
    report.append("=" * 120)
    report.append("报告生成完成")
    report.append("=" * 120)
    
    return '\n'.join(report)


# ==================== 9. 保存结果 ====================
def save_results(results, trained_models, importance_results, shap_results):
    """保存所有分析结果到文件"""
    print("\n" + "=" * 100)
    print("步骤 7: 保存分析结果")
    print("=" * 100)
    
    # 1. 保存模型性能汇总
    print("\n保存模型性能汇总...")
    performance_data = []
    for model_name, model_results in results.items():
        performance_data.append({
            '模型': model_name,
            '训练集R²': model_results['train_r2'],
            '测试集R²': model_results['test_r2'],
            '训练集RMSE': model_results['train_rmse'],
            '测试集RMSE': model_results['test_rmse'],
            '训练集MAE': model_results['train_mae'],
            '测试集MAE': model_results['test_mae'],
            '测试集MAPE': model_results['test_mape'],
            'CV_R²_均值': model_results['cv_r2_mean'],
            'CV_R²_标准差': model_results['cv_r2_std'],
            '过拟合差距': model_results['overfit_gap']
        })
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values('测试集R²', ascending=False)
    performance_df.to_csv(out_path('model_performance_summary.csv'), index=False, encoding='utf-8-sig')
    print("✓ 保存: model_performance_summary.csv")
    
    # 2. 保存特征重要性汇总
    if importance_results:
        print("\n保存特征重要性汇总...")
        with pd.ExcelWriter(out_path('feature_importance_summary.xlsx'), engine='openpyxl') as writer:
            for model_name, importance_series in importance_results.items():
                if importance_series is not None:
                    df_imp = pd.DataFrame({
                        '特征': importance_series.index,
                        '重要性': importance_series.values
                    })
                    sheet_name = model_name[:31]  # Excel sheet名称限制
                    df_imp.to_excel(writer, sheet_name=sheet_name, index=False)
        print("✓ 保存: feature_importance_summary.xlsx")
    
    # 3. 保存SHAP重要性汇总
    if shap_results:
        print("\n保存SHAP重要性汇总...")
        with pd.ExcelWriter(out_path('shap_importance_summary.xlsx'), engine='openpyxl') as writer:
            for model_name, shap_data in shap_results.items():
                if 'importance' in shap_data and shap_data['importance'] is not None:
                    df_shap = shap_data['importance']
                    sheet_name = model_name[:31]
                    df_shap.to_excel(writer, sheet_name=sheet_name, index=False)
        print("✓ 保存: shap_importance_summary.xlsx")
    
    # 4. 保存最佳超参数
    print("\n保存最佳超参数配置...")
    hyperparams_data = []
    for model_name, model_results in results.items():
        params_str = '; '.join([f"{k}={v}" for k, v in model_results['best_params'].items()])
        hyperparams_data.append({
            '模型': model_name,
            '测试集R²': model_results['test_r2'],
            '最佳超参数': params_str
        })
    
    hyperparams_df = pd.DataFrame(hyperparams_data)
    hyperparams_df = hyperparams_df.sort_values('测试集R²', ascending=False)
    hyperparams_df.to_csv(out_path('best_hyperparameters.csv'), index=False, encoding='utf-8-sig')
    print("✓ 保存: best_hyperparameters.csv")
    
    # 5. 保存训练好的模型
    print("\n保存训练好的模型...")
    import pickle
    for model_name, model_info in trained_models.items():
        model_filename = f"model_{model_name.replace(' ', '_')}.pkl"
        with open(out_path(model_filename), 'wb') as f:
            pickle.dump(model_info['model'], f)
        print(f"✓ 保存: {model_filename}")
    
    # 6. 保存预测结果对比
    print("\n保存预测结果对比...")
    first_model = list(results.keys())[0]
    pred_length = len(results[first_model]['y_pred_test'])
    predictions_data = {'样本序号': range(1, pred_length + 1)}
    
    for model_name, model_results in results.items():
        if 'y_pred_test' in model_results:
            predictions_data[f'{model_name}_预测值'] = model_results['y_pred_test']
    
    pred_df = pd.DataFrame(predictions_data)
    pred_df.to_csv(out_path('predictions_comparison.csv'), index=False, encoding='utf-8-sig')
    print("✓ 保存: predictions_comparison.csv")
    print("  (注: 包含真实值的完整残差数据已保存在 plot_data/PlotData_Prediction_Residuals.csv 中)")
    
    print("\n所有结果保存完成！")  

# ==================== 导出绘图矩阵数据的函数  ====================
def export_plot_raw_data(results, shap_results, y_test):
    """提取核心指标和数组，导出为结构化的CSV宽表，方便 Origin/Excel 重新作图"""
    plot_data_dir = os.path.join(OUTPUT_DIR, "plot_data")
    os.makedirs(plot_data_dir, exist_ok=True)
    
    print("\n" + "=" * 100)
    print("正在导出所有图形的原始数据到 plot_data 文件夹...")
    
    # 1. 预测值 vs 真实值 vs 残差矩阵
    try:
        pred_res_data = {'Actual_PM': y_test.values}
        for m, r in results.items():
            if 'y_pred_test' in r:
                y_pred = r['y_pred_test']
                # 确保维度一致
                if len(y_pred) == len(y_test):
                    pred_res_data[f'{m}_Predicted'] = y_pred
                    pred_res_data[f'{m}_Residual'] = y_test.values - y_pred
                    safe_y = np.where(y_test.values == 0, 1e-5, y_test.values)
                    pred_res_data[f'{m}_RelError_pct'] = np.abs((y_test.values - y_pred) / safe_y) * 100
                
        df_pred = pd.DataFrame(pred_res_data, index=y_test.index)
        df_pred.to_csv(os.path.join(plot_data_dir, 'PlotData_Prediction_Residuals.csv'), index=False, encoding='utf-8-sig')
        print("  ✓ 预测值与残差矩阵导出成功")
    except Exception as e:
        print(f"  ✗ 预测值导出失败: {e}")

    # 2. 强制导出 Soft-Voting Ensemble 的 SHAP 散点图原始矩阵
    target_model = 'Soft-Voting Ensemble'
    
    # 如果没有 Ensemble，就退而求其次选 R2 最高的基础模型
    if target_model not in shap_results:
        print(f"  ⚠ 未找到 {target_model} 的 SHAP 数据，寻找最佳替代模型...")
        if results and shap_results:
            target_model = max([k for k in results.keys() if k in shap_results], 
                               key=lambda k: results[k]['test_r2'], default=None)

    if target_model and target_model in shap_results:
        try:
            print(f"  → 正在提取 [{target_model}] 的 SHAP 绘图矩阵...")
            sd = shap_results[target_model]
            sv = sd['shap_values']              # 这是一个 numpy array
            X_samp = sd['X_sample']             # 可能是 DataFrame 也可能是 numpy array
            features = sd['feature_names_used'] # 特征名列表
            
            # 提取所有特征（为了能在 Origin 里画完整的图，直接导出所有特征）
            top_feats = sd['importance']['Feature'].tolist()
            
            shap_scatter_dict = {}
            for feat in top_feats:
                if feat in features:
                    idx = features.index(feat)
                    
                    # 极其严谨的取值方式，防止 Pandas/Numpy 混用导致的报错
                    if isinstance(X_samp, pd.DataFrame):
                        f_val = X_samp[feat].values
                    else:
                        f_val = X_samp[:, idx]
                        
                    # 确保维度展平为 1D 数组
                    shap_scatter_dict[f'{feat}_FeatureValue'] = np.ravel(f_val)
                    shap_scatter_dict[f'{feat}_SHAPValue'] = np.ravel(sv[:, idx])
            
            # 转换为 DataFrame 并导出
            df_shap_scatter = pd.DataFrame(shap_scatter_dict)
            safe_name = target_model.replace(" ", "_")
            file_path = os.path.join(plot_data_dir, f'PlotData_SHAP_Scatter_{safe_name}.csv')
            df_shap_scatter.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            print(f"  ✓ [{target_model}] SHAP 散点矩阵导出成功！文件: PlotData_SHAP_Scatter_{safe_name}.csv")
            print(f"    (包含 {len(top_feats)} 个特征的 SHAP 值及其对应的原始特征值)")
            
        except Exception as e:
            print(f"  ✗ [{target_model}] SHAP 散点矩阵导出惨遭失败: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"  ✗ 无法导出 SHAP 散点矩阵：{target_model} 不在 shap_results 中。")
        
    print(f"✓ 绘图原始数据流程执行完毕！")

# ==================== 10. SHAP-based R² 分析（新增核心模块）====================
def compute_shap_r2(shap_results, results, y_test, feature_names):

    print("\n" + "=" * 100)
    print("步骤 8: SHAP-based R² 分析（特征对因变量方差的百分比贡献）")
    print("=" * 100)

    shap_r2_all = {}
    y_var = np.var(y_test.values, ddof=1)   # 测试集目标变量方差（无偏估计）

    print(f"\n测试集 Metro_PM 总方差: {y_var:.4f}  (std={np.std(y_test.values, ddof=1):.4f})")

    for model_name, shap_data in shap_results.items():
        print(f"\n{'─'*80}")
        print(f" 模型: {model_name}")

        # ── 1. 获取 SHAP 矩阵与特征名 ────────────────────────────────────────
        sv              = shap_data['shap_values']          # shape: (n_samples, n_features)
        feat_names_used = shap_data['feature_names_used']   # 与 sv 列对齐的特征名

        if sv is None or sv.ndim != 2:
            print(f"  ✗ SHAP 值维度异常，跳过")
            continue

        n_samples, n_feats = sv.shape
        model_r2 = results.get(model_name, {}).get('test_r2', None)

        if model_r2 is None:
            print(f"  ✗ 找不到模型 R²，跳过")
            continue

        print(f"  样本数={n_samples} | 特征数={n_feats} | 模型测试R²={model_r2:.4f}")

        # ── 2. 计算每个特征的 SHAP 方差 ──────────────────────────────────────
        shap_var = np.var(sv, axis=0, ddof=1)              # shape: (n_feats,)
        total_shap_var = shap_var.sum()

        if total_shap_var < 1e-10:
            print(f"  ✗ SHAP 方差总和趋近于0，跳过")
            continue

        # ── 3. 计算 SHAP-based R²（各特征对模型 R² 的分配）─────────────────
        #   原理：各特征 SHAP 方差占比 × 模型 R²
        #   保证：Σ R²_j = R²_model（精确守恒）
        shap_r2_per_feat = (shap_var / total_shap_var) * model_r2   # shape: (n_feats,)
        shap_r2_pct      = shap_r2_per_feat / model_r2 * 100        # 百分比（相对于R²）

        # ── 4. 计算方向（SHAP 均值正负）────────────────────────────────────
        mean_shap = sv.mean(axis=0)
        directions = ['正影响(↑)' if v >= 0 else '负影响(↓)' for v in mean_shap]

        # ── 5. 组装 DataFrame ────────────────────────────────────────────────
        df_r2 = pd.DataFrame({
            'Feature':       feat_names_used,
            'SHAP_Var':      shap_var,
            'Mean_SHAP':     mean_shap,
            'SHAP_R2':       shap_r2_per_feat,
            'SHAP_R2_Pct':   shap_r2_pct,
            'Direction':     directions,
        }).sort_values('SHAP_R2', ascending=False).reset_index(drop=True)

        # 累积百分比
        df_r2['Cumulative_Pct'] = df_r2['SHAP_R2_Pct'].cumsum()

        # ── 6. 验证守恒性 ────────────────────────────────────────────────────
        r2_sum   = df_r2['SHAP_R2'].sum()
        r2_check = abs(r2_sum - model_r2) < 1e-8
        print(f" ✅ 守恒验证: ∑ SHAP-R²={r2_sum:.6f} 模型 R²={model_r2:.6f} "
            f"{'✅ 一致' if r2_check else f'⚠️ 误差 > {abs(r2_sum-model_r2):.2e}'}")

        shap_r2_all[model_name] = df_r2

        # ── 7. 打印 Top-15 结果 ──────────────────────────────────────────────
        print(f"\n  {'特征':<30} {'SHAP方差':>10} {'SHAP-R²':>10} "
              f"{'占R²%':>9} {'累积%':>9} {'方向'}")
        print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*9} {'─'*9} {'─'*10}")

        for _, row in df_r2.head(15).iterrows():
            bar_len = int(row['SHAP_R2_Pct'] / 100 * 25)   # 25格宽进度条
            bar     = '█' * bar_len + '░' * (25 - bar_len)
            print(f"  {row['Feature']:<30} {row['SHAP_Var']:>10.5f} "
                  f"{row['SHAP_R2']:>10.5f} {row['SHAP_R2_Pct']:>8.2f}% "
                  f"{row['Cumulative_Pct']:>8.2f}% {row['Direction']:<12}  {bar}")

        # 打印汇总
        top5_sum  = df_r2.head(5)['SHAP_R2_Pct'].sum()
        top10_sum = df_r2.head(10)['SHAP_R2_Pct'].sum()
        print(f"\n  📊 Top-5  特征累计贡献: {top5_sum:.2f}%")
        print(f"  📊 Top-10 特征累计贡献: {top10_sum:.2f}%")
        print(f"  📊 全部特征总贡献=模型R²: {r2_sum:.4f} ({r2_sum*100:.2f}%)")

    success = len(shap_r2_all)
    print(f"\nSHAP-based R² 计算完成: {success}/{len(shap_results)} 个模型")
    return shap_r2_all


# ==================== 11. SHAP-R² 可视化====================
def visualize_shap_r2(shap_r2_all, results):
    if not shap_r2_all:
        print("⚠ 无 SHAP-R² 数据，跳过可视化")
        return

    print("\n生成 SHAP-based R² 可视化图表...")

    # ── 找最优模型 ────────────────────────────────────────────────────────────
    best_model = max(
        [m for m in shap_r2_all if m in results],
        key=lambda m: results[m]['test_r2']
    )
    print(f"  最优模型（用于详细图）: {best_model} (R²={results[best_model]['test_r2']:.4f})")
    print("  生成图(a)：各模型 Top-15 SHAP-R² 对比图...")

    # 收集所有模型的 Top-15 特征，取并集作为公共行
    all_top_feats = []
    for df_r2 in shap_r2_all.values():
        all_top_feats.extend(df_r2.head(15)['Feature'].tolist())
    from collections import Counter
    top_feats_union = [f for f, _ in Counter(all_top_feats).most_common(18)]

    model_list = list(shap_r2_all.keys())
    n_models   = len(model_list)
    n_top      = min(15, len(top_feats_union))

    fig, axes_a = plt.subplots(
        1, n_models,
        figsize=(min(7 * n_models, 42), max(8, n_top * 0.6 + 2)),
        sharey=True
    )
    if n_models == 1:
        axes_a = [axes_a]

    cmap_pos = plt.cm.Reds
    cmap_neg = plt.cm.Blues

    for mi, (mname, ax_a) in enumerate(zip(model_list, axes_a)):
        df_r2  = shap_r2_all[mname]
        imp    = df_r2.set_index('Feature')
        values = []
        colors_bar_a = []
        for f in top_feats_union[:n_top]:
            row = imp.loc[f] if f in imp.index else None
            val = float(row['SHAP_R2_Pct']) if row is not None else 0.0
            dir_= row['Direction'] if row is not None else '正影响(↑)'
            values.append(val)
            colors_bar_a.append('#e74c3c' if dir_ == '正影响(↑)' else '#3498db')

        y_pos = np.arange(n_top)
        bars_a = ax_a.barh(y_pos, values, color=colors_bar_a, alpha=0.82, edgecolor='white', linewidth=0.5)
        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels(top_feats_union[:n_top], fontsize=9)
        ax_a.invert_yaxis()
        ax_a.set_xlabel('SHAP-R² 贡献 (%)', fontsize=10)
        model_r2 = results.get(mname, {}).get('test_r2', 0)
        ax_a.set_title(f'{mname}\n(模型R²={model_r2:.4f})', fontsize=11, fontweight='bold')
        ax_a.grid(axis='x', alpha=0.3, linestyle='--')
        ax_a.axvline(0, color='gray', linewidth=0.8)

        # 添加数值标签
        for bar_a, val in zip(bars_a, values):
            if val > 0.5:  # 只标注贡献>0.5%的特征
                ax_a.text(
                    bar_a.get_width() + 0.3, bar_a.get_y() + bar_a.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=7, color='black'
                )

    # 图例
    from matplotlib.patches import Patch
    legend_a = [
        Patch(facecolor='#e74c3c', label='正影响（增大PM）'),
        Patch(facecolor='#3498db', label='负影响（减小PM）')
    ]
    fig.legend(handles=legend_a, loc='upper center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.98), frameon=False)

    plt.suptitle('各模型 SHAP-based R² 特征贡献对比（Top-15）',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path('shap_r2_comparison_all_models.png'), dpi=300, bbox_inches='tight')
    print("  ✓ 保存: shap_r2_comparison_all_models.png")
    plt.close()

    print("  生成图(b)：最优模型 SHAP-R² 瀑布图...")

    df_best = shap_r2_all[best_model].head(20)
    n_show  = len(df_best)

    fig_b, ax_b = plt.subplots(figsize=(14, max(10, n_show * 0.5)))

    # 计算累积位置（瀑布效果）
    cumsum_vals = np.concatenate([[0], df_best['SHAP_R2_Pct'].cumsum().values])
    starts      = cumsum_vals[:-1]
    heights     = df_best['SHAP_R2_Pct'].values

    colors_b = ['#e74c3c' if d == '正影响(↑)' else '#3498db'
                for d in df_best['Direction']]

    y_pos_b = np.arange(n_show)
    bars_b  = ax_b.barh(y_pos_b, heights, left=starts, color=colors_b,
                        alpha=0.85, edgecolor='white', linewidth=1.2)

    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(df_best['Feature'], fontsize=10)
    ax_b.invert_yaxis()
    ax_b.set_xlabel('累积 SHAP-R² 贡献 (%)', fontsize=12)
    ax_b.set_title(f'{best_model} — SHAP-based R² 瀑布图（Top-20特征）\n'
                   f'模型总R²={results[best_model]["test_r2"]:.4f}',
                   fontsize=13, fontweight='bold', pad=15)
    ax_b.grid(axis='x', alpha=0.3)

    # 标注每个条形的贡献值
    for i, (bar_b, h, s) in enumerate(zip(bars_b, heights, starts)):
        mid_x = s + h / 2
        ax_b.text(mid_x, bar_b.get_y() + bar_b.get_height()/2,
                  f'{h:.2f}%', ha='center', va='center',
                  fontsize=8, fontweight='bold', color='white')

    # 添加总R²标注线
    total_r2_pct = df_best['SHAP_R2_Pct'].sum()
    ax_b.axvline(total_r2_pct, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_b.text(total_r2_pct + 1, n_show * 0.5,
              f'Top-20累积\n{total_r2_pct:.1f}%',
              fontsize=10, color='red', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    ax_b.legend(handles=legend_a, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path(f'shap_r2_waterfall_{best_model.replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: shap_r2_waterfall_{best_model.replace(' ', '_')}.png")
    plt.close()

    if len(shap_r2_all) > 1:
        print("  生成图(c)：跨模型 SHAP-R² 热力图...")

        # 构建矩阵：行=特征，列=模型
        all_feats_c = set()
        for df_r2 in shap_r2_all.values():
            all_feats_c.update(df_r2['Feature'].tolist())

        r2_matrix = pd.DataFrame(index=sorted(all_feats_c), columns=model_list)

        for mname, df_r2 in shap_r2_all.items():
            for _, row in df_r2.iterrows():
                r2_matrix.loc[row['Feature'], mname] = row['SHAP_R2_Pct']

        r2_matrix = r2_matrix.fillna(0).astype(float)

        # 选择 Top-25 特征（按总贡献排序）
        top_feats_c = r2_matrix.sum(axis=1).nlargest(25).index

        fig_c, ax_c = plt.subplots(figsize=(max(12, len(model_list) * 1.5), 14))
        sns.heatmap(
            r2_matrix.loc[top_feats_c],
            annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'SHAP-R² 贡献 (%)'},
            linewidths=0.5, linecolor='white',
            ax=ax_c, vmin=0, vmax=r2_matrix.loc[top_feats_c].max().max()
        )
        ax_c.set_title('跨模型 SHAP-based R² 热力图（Top-25特征）',
                       fontsize=14, fontweight='bold', pad=15)
        ax_c.set_xlabel('模型', fontsize=12)
        ax_c.set_ylabel('特征', fontsize=12)
        ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=45, ha='right')
        ax_c.set_yticklabels(ax_c.get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(out_path('shap_r2_heatmap_cross_models.png'), dpi=300, bbox_inches='tight')
        print("  ✓ 保存: shap_r2_heatmap_cross_models.png")
        plt.close()

    print("  生成图(d)：SHAP-R² 累积贡献曲线（帕累托图）...")

    fig_d, axes_d = plt.subplots(1, min(3, n_models), figsize=(18, 6))
    if n_models == 1:
        axes_d = [axes_d]
    elif n_models == 2:
        axes_d = list(axes_d)
    else:
        axes_d = axes_d.flatten()

    # 选择 Top-3 模型（按 R² 排序）
    top3_models = sorted(
        [m for m in shap_r2_all if m in results],
        key=lambda m: results[m]['test_r2'], reverse=True
    )[:min(3, n_models)]

    for idx_d, mname_d in enumerate(top3_models):
        df_d = shap_r2_all[mname_d].head(30)
        ax_d = axes_d[idx_d]

        x_d = np.arange(1, len(df_d) + 1)
        y_d = df_d['Cumulative_Pct'].values

        # 双轴：左轴=累积%，右轴=单特征贡献%
        ax_d2 = ax_d.twinx()

        # 累积曲线
        ax_d.plot(x_d, y_d, marker='o', linewidth=2.5, color='#2c3e50',
                  markersize=6, markerfacecolor='#e74c3c', markeredgewidth=1.5,
                  markeredgecolor='white', label='累积贡献')
        ax_d.fill_between(x_d, 0, y_d, alpha=0.2, color='#3498db')

        # 单特征贡献条形图
        bars_d = ax_d2.bar(x_d, df_d['SHAP_R2_Pct'].values, alpha=0.5,
                           color='#95a5a6', edgecolor='white', linewidth=0.8,
                           label='单特征贡献')

        # 80% 参考线（帕累托法则）
        ax_d.axhline(80, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        n_80 = (y_d >= 80).argmax() + 1 if (y_d >= 80).any() else len(df_d)
        ax_d.text(len(df_d) * 0.7, 82, f'80%线（前{n_80}特征）',
                  fontsize=9, color='red', fontweight='bold')

        ax_d.set_xlabel('特征排名（按 SHAP-R² 降序）', fontsize=11)
        ax_d.set_ylabel('累积贡献 (%)', fontsize=11, color='#2c3e50')
        ax_d2.set_ylabel('单特征贡献 (%)', fontsize=11, color='#95a5a6')
        ax_d.set_title(f'{mname_d}\n(R²={results[mname_d]["test_r2"]:.4f})',
                       fontsize=12, fontweight='bold')
        ax_d.set_xlim(0.5, len(df_d) + 0.5)
        ax_d.set_ylim(0, 105)
        ax_d.grid(axis='y', alpha=0.3, linestyle=':')
        ax_d.tick_params(axis='y', labelcolor='#2c3e50')
        ax_d2.tick_params(axis='y', labelcolor='#95a5a6')

        # 图例
        lines_d, labels_d = ax_d.get_legend_handles_labels()
        lines_d2, labels_d2 = ax_d2.get_legend_handles_labels()
        ax_d.legend(lines_d + lines_d2, labels_d + labels_d2,
                    loc='lower right', fontsize=9)

    # 隐藏多余子图
    for idx_d in range(len(top3_models), len(axes_d)):
        axes_d[idx_d].axis('off')

    plt.suptitle('SHAP-based R² 累积贡献曲线（帕累托分析）',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(out_path('shap_r2_pareto_curves.png'), dpi=300, bbox_inches='tight')
    print("  ✓ 保存: shap_r2_pareto_curves.png")
    plt.close()

    print("  生成图(e)：正负影响分组双向条形图...")

    df_e = shap_r2_all[best_model].head(20).copy()
    df_pos = df_e[df_e['Direction'] == '正影响(↑)'].copy()
    df_neg = df_e[df_e['Direction'] == '负影响(↓)'].copy()

    fig_e, ax_e = plt.subplots(figsize=(14, max(10, len(df_e) * 0.5)))

    # 合并并按绝对值排序
    df_e['SHAP_R2_Signed'] = df_e.apply(
        lambda row: row['SHAP_R2_Pct'] if row['Direction'] == '正影响(↑)' 
        else -row['SHAP_R2_Pct'], axis=1
    )
    df_e_sorted = df_e.sort_values('SHAP_R2_Signed', ascending=True)

    y_pos_e = np.arange(len(df_e_sorted))
    colors_e = ['#e74c3c' if v > 0 else '#3498db' 
                for v in df_e_sorted['SHAP_R2_Signed']]

    bars_e = ax_e.barh(y_pos_e, df_e_sorted['SHAP_R2_Signed'].values,
                       color=colors_e, alpha=0.85, edgecolor='white', linewidth=1)

    ax_e.set_yticks(y_pos_e)
    ax_e.set_yticklabels(df_e_sorted['Feature'], fontsize=10)
    ax_e.set_xlabel('SHAP-R² 贡献 (%) [正=增大PM, 负=减小PM]', fontsize=12)
    ax_e.set_title(f'{best_model} — SHAP-based R² 正负影响分组图\n'
                   f'模型R²={results[best_model]["test_r2"]:.4f}',
                   fontsize=13, fontweight='bold', pad=15)
    ax_e.axvline(0, color='black', linewidth=1.5)
    ax_e.grid(axis='x', alpha=0.3, linestyle='--')

    # 标注数值
    for bar_e, val in zip(bars_e, df_e_sorted['SHAP_R2_Signed'].values):
        x_pos = val + (0.5 if val > 0 else -0.5)
        ax_e.text(x_pos, bar_e.get_y() + bar_e.get_height()/2,
                  f'{abs(val):.2f}%', ha='left' if val > 0 else 'right',
                  va='center', fontsize=8, fontweight='bold')

    # 统计信息
    pos_sum = df_pos['SHAP_R2_Pct'].sum()
    neg_sum = df_neg['SHAP_R2_Pct'].sum()
    ax_e.text(0.98, 0.98, f'正影响总计: {pos_sum:.2f}%\n负影响总计: {neg_sum:.2f}%',
              transform=ax_e.transAxes, fontsize=10, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.6))

    ax_e.legend(handles=legend_a, loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path(f'shap_r2_bidirectional_{best_model.replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: shap_r2_bidirectional_{best_model.replace(' ', '_')}.png")
    plt.close()

    print("\n所有 SHAP-based R² 可视化图表生成完成！")


# ==================== 12. 保存 SHAP-R² 结果====================
def save_shap_r2_results(shap_r2_all):
    """保存 SHAP-based R² 分析结果到 Excel 和 CSV"""
    if not shap_r2_all:
        return

    print("\n保存 SHAP-based R² 结果...")

    # ── 1. 保存到 Excel（每个模型一个 sheet）────────────────────────────
    excel_path = out_path('shap_r2_analysis.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for model_name, df_r2 in shap_r2_all.items():
            sheet_name = model_name[:31]  # Excel sheet 名称限制
            df_r2.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  ✓ 保存: shap_r2_analysis.xlsx")

    # ── 2. 保存汇总 CSV（所有模型的 Top-20）──────────────────────────────
    summary_rows = []
    for model_name, df_r2 in shap_r2_all.items():
        for _, row in df_r2.head(20).iterrows():
            summary_rows.append({
                '模型': model_name,
                '特征': row['Feature'],
                'SHAP方差': row['SHAP_Var'],
                'SHAP均值': row['Mean_SHAP'],
                'SHAP_R²': row['SHAP_R2'],
                'SHAP_R²百分比': row['SHAP_R2_Pct'],
                '累积百分比': row['Cumulative_Pct'],
                '影响方向': row['Direction']
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path('shap_r2_summary_top20.csv'), index=False, encoding='utf-8-sig')
    print(f"  ✓ 保存: shap_r2_summary_top20.csv")

    # ── 3. 保存跨模型对比矩阵（特征×模型）────────────────────────────────
    if len(shap_r2_all) > 1:
        all_features = set()
        for df_r2 in shap_r2_all.values():
            all_features.update(df_r2['Feature'].tolist())

        comparison_matrix = pd.DataFrame(
            index=sorted(all_features),
            columns=list(shap_r2_all.keys())
        )

        for model_name, df_r2 in shap_r2_all.items():
            for _, row in df_r2.iterrows():
                comparison_matrix.loc[row['Feature'], model_name] = row['SHAP_R2_Pct']

        comparison_matrix = comparison_matrix.fillna(0)
        comparison_matrix['平均贡献'] = comparison_matrix.mean(axis=1)
        comparison_matrix = comparison_matrix.sort_values('平均贡献', ascending=False)

        comparison_matrix.to_csv(
            out_path('shap_r2_cross_model_comparison.csv'),
            encoding='utf-8-sig'
        )
        print(f"  ✓ 保存: shap_r2_cross_model_comparison.csv")

    print("SHAP-based R² 结果保存完成！")


# ==================== 13. 更新报告生成函数====================
def append_shap_r2_to_report(report_text, shap_r2_all, results):
    """在原有报告基础上追加 SHAP-based R² 章节"""
    if not shap_r2_all:
        return report_text

    report = report_text.split('\n')
    
    # 找到插入位置（在"九、结论与建议"之前）
    insert_idx = None
    for i, line in enumerate(report):
        if line.startswith("九、结论与建议"):
            insert_idx = i
            break
    
    if insert_idx is None:
        insert_idx = len(report) - 10  

    # 构建新章节
    new_section = []
    new_section.append("九、SHAP-based R² 分析（特征对因变量方差的解释贡献）")
    new_section.append("-" * 120)
    new_section.append("")
    new_section.append("【方法说明】")
    new_section.append("SHAP-based R² 量化了每个自变量对模型 R² 的贡献百分比，计算公式为：")
    new_section.append("  R²_j = [Var(φⱼ) / Σₖ Var(φₖ)] × R²_model")
    new_section.append("其中 Var(φⱼ) 是特征 j 的 SHAP 值在所有样本上的方差。")
    new_section.append("该方法确保：Σ R²_j = R²_model（精确守恒）")
    new_section.append("")

    for model_name, df_r2 in shap_r2_all.items():
        model_r2 = results.get(model_name, {}).get('test_r2', 0)
        new_section.append(f"\n【{model_name}】（模型R²={model_r2:.4f}）")
        new_section.append("")
        new_section.append("Top 10 特征对 R² 的贡献：")
        new_section.append(f"{'排名':<6} {'特征':<30} {'SHAP-R²':>10} {'占R²%':>10} "
                          f"{'累积%':>10} {'方向':<12}")
        new_section.append("-" * 90)

        for rank, (_, row) in enumerate(df_r2.head(10).iterrows(), 1):
            new_section.append(
                f"{rank:<6} {row['Feature']:<30} {row['SHAP_R2']:>10.5f} "
                f"{row['SHAP_R2_Pct']:>9.2f}% {row['Cumulative_Pct']:>9.2f}% "
                f"{row['Direction']:<12}"
            )

        # 统计摘要
        top5_sum = df_r2.head(5)['SHAP_R2_Pct'].sum()
        top10_sum = df_r2.head(10)['SHAP_R2_Pct'].sum()
        pos_feats = df_r2[df_r2['Direction'] == '正影响(↑)']
        neg_feats = df_r2[df_r2['Direction'] == '负影响(↓)']

        new_section.append("")
        new_section.append(f"统计摘要：")
        new_section.append(f"  • Top-5  特征累计贡献: {top5_sum:.2f}%")
        new_section.append(f"  • Top-10 特征累计贡献: {top10_sum:.2f}%")
        new_section.append(f"  • 正影响特征数量: {len(pos_feats)} 个，总贡献: {pos_feats['SHAP_R2_Pct'].sum():.2f}%")
        new_section.append(f"  • 负影响特征数量: {len(neg_feats)} 个，总贡献: {neg_feats['SHAP_R2_Pct'].sum():.2f}%")
        new_section.append("")

    # 跨模型对比
    if len(shap_r2_all) > 1:
        new_section.append("\n【跨模型特征贡献对比】")
        new_section.append("")

        # 找出在所有模型中都排 Top-10 的特征
        from collections import Counter
        all_top10 = []
        for df_r2 in shap_r2_all.values():
            all_top10.extend(df_r2.head(10)['Feature'].tolist())
        
        top10_counter = Counter(all_top10)
        consistent_feats = [f for f, cnt in top10_counter.most_common(10) 
                           if cnt >= len(shap_r2_all) * 0.5]  # 至少在50%模型中排前10

        if consistent_feats:
            new_section.append("在多数模型中稳定排名 Top-10 的关键特征：")
            for feat in consistent_feats:
                avg_contrib = np.mean([
                    df_r2[df_r2['Feature'] == feat]['SHAP_R2_Pct'].values[0]
                    if feat in df_r2['Feature'].values else 0
                    for df_r2 in shap_r2_all.values()
                ])
                new_section.append(f"  • {feat:<30} 平均贡献: {avg_contrib:.2f}%")
        
        new_section.append("")

    # 关键发现
    new_section.append("\n【关键发现】")
    new_section.append("")

    # 找最优模型的最重要特征
    best_model = max(
        [m for m in shap_r2_all if m in results],
        key=lambda m: results[m]['test_r2']
    )
    df_best = shap_r2_all[best_model]
    top1_feat = df_best.iloc[0]

    new_section.append(f"1. 最关键特征（{best_model}）：")
    new_section.append(f"   {top1_feat['Feature']} 单独解释了 {top1_feat['SHAP_R2_Pct']:.2f}% 的模型R²")
    new_section.append(f"   影响方向：{top1_feat['Direction']}")
    new_section.append("")

    # 帕累托分析
    cumsum_80 = (df_best['Cumulative_Pct'] >= 80).idxmax()
    n_80 = cumsum_80 + 1 if cumsum_80 >= 0 else len(df_best)
    new_section.append(f"2. 帕累托法则验证：")
    new_section.append(f"   前 {n_80} 个特征累计贡献达到 80% 的模型R²")
    new_section.append(f"   占总特征数的 {n_80/len(df_best)*100:.1f}%")
    new_section.append("")

    # 正负影响对比
    pos_sum = df_best[df_best['Direction'] == '正影响(↑)']['SHAP_R2_Pct'].sum()
    neg_sum = df_best[df_best['Direction'] == '负影响(↓)']['SHAP_R2_Pct'].sum()
    new_section.append(f"3. 正负影响平衡：")
    new_section.append(f"   正影响特征（增大PM）总贡献: {pos_sum:.2f}%")
    new_section.append(f"   负影响特征（减小PM）总贡献: {neg_sum:.2f}%")
    if pos_sum > neg_sum * 1.5:
        new_section.append(f"   ⚠ 正影响特征主导，建议重点控制高贡献的正影响因素")
    elif neg_sum > pos_sum * 1.5:
        new_section.append(f"   ✓ 负影响特征主导，说明存在有效的PM抑制因素")
    else:
        new_section.append(f"   ○ 正负影响相对平衡")
    new_section.append("")

    # 插入新章节
    report[insert_idx:insert_idx] = new_section
    
    # 更新后续章节编号（九→十，等等）
    for i in range(insert_idx + len(new_section), len(report)):
        if report[i].startswith("九、"):
            report[i] = report[i].replace("九、", "十、", 1)
        elif report[i].startswith("【"):
            continue  # 保持子标题不变

    return '\n'.join(report)

def main(file_path='2.xlsx'):
    """主执行函数（已集成 SHAP-based R² 分析）"""
    print("\n开始分析流程...\n")
    try:
        # 步骤1-7: 原有流程（保持不变）
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names = \
            load_and_preprocess_data(file_path)
        
        if X_train_scaled is None or X_test_scaled is None:
            print("警告: X_train_scaled和X_test_scaled未定义，使用原始数据进行标准化")
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
        
        USE_AUGMENT = False
        N_AUG = 1
        NOISE_SCALE = 0.1
        if USE_AUGMENT:
            print("\n>>> 启用残差驱动数据增强")
            print(f" 原始训练样本数: {len(X_train)}")
            X_aug, y_aug = residual_based_augmentation(
                X_train, y_train, n_aug=N_AUG, noise_scale=NOISE_SCALE
            )
            X_train = pd.concat([X_train, X_aug], axis=0)
            y_train = pd.concat([y_train, y_aug], axis=0)
            print(f" 增强后训练样本数: {len(X_train)}")
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
        
        results, trained_models = train_and_evaluate_models(
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
        )
        
        importance_results = analyze_feature_importance(
            trained_models,results, X_train, X_test, y_train, y_test, feature_names
        )
        
        shap_results = analyze_shap_values(
            trained_models, results, X_train, X_test,
            X_train_scaled, X_test_scaled, feature_names, y_train, y_test
        )

        create_visualizations(results, importance_results, shap_results, 
                             feature_names, y_test)
        
        shap_r2_all = compute_shap_r2(shap_results, results, y_test, feature_names)
        
        visualize_shap_r2(shap_r2_all, results)
    
        save_shap_r2_results(shap_r2_all)

        export_plot_raw_data(results, shap_results, y_test)

        base_report = generate_report(results, importance_results, shap_results, feature_names)
        
        enhanced_report = append_shap_r2_to_report(base_report, shap_r2_all, results)
        
        with open(out_path('analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write(enhanced_report)
        print(f"\n✓ 增强版报告已保存（含 SHAP-based R² 章节）")

        save_results(results, trained_models, importance_results, shap_results)
        
        print("\n" + "=" * 100)
        print("分析完成！")
        print("=" * 100)
        print(f"\n📁 所有输出文件已保存至目录: {OUTPUT_DIR}")
        print("\n生成的文件清单:")
        print("\n【可视化图表】")
        print("  1.  model_performance_comparison.png    - 模型性能对比图")
        print("  2.  prediction_vs_actual.png            - 预测值vs真实值散点图")
        print("  3.  residual_analysis.png               - 残差分析图")
        print("  4.  feature_importance.png              - 特征重要性图")
        print("  5.  feature_importance_heatmap.png      - 特征重要性热力图")
        print("  6.  model_radar_chart.png               - 模型性能雷达图")
        print("  7.  error_distribution.png              - 误差分布箱线图")
        print("  8.  overfitting_diagnosis.png           - 过拟合诊断图")
        print("  9.  shap_importance_heatmap.png         - SHAP重要性热力图")
        
        print("\n【SHAP可视化】(每个模型)")
        for model_name in shap_results.keys():
            model_file = model_name.replace(' ', '_')
            print(f"  -  shap_summary_{model_file}.png        - SHAP摘要图")
            print(f"  -  shap_bar_{model_file}.png            - SHAP条形图")
            print(f"  -  shap_waterfall_{model_file}.png      - SHAP瀑布图")
        
        print("\n【★ SHAP-based R² 可视化（新增）】")
        print("  10. shap_r2_comparison_all_models.png   - 各模型 SHAP-R² 对比图")
        print("  11. shap_r2_waterfall_[best_model].png  - 最优模型瀑布图")
        print("  12. shap_r2_heatmap_cross_models.png    - 跨模型热力图")
        print("  13. shap_r2_pareto_curves.png           - 累积贡献曲线（帕累托）")
        print("  14. shap_r2_bidirectional_[best].png    - 正负影响双向图")
        
        print("\n【分析报告】")
        print("  15. analysis_report.txt                 - 详细分析报告（含SHAP-R²章节）")
        
        print("\n【数据文件】")
        print("  16. model_performance_summary.csv       - 模型性能汇总")
        print("  17. feature_importance_summary.xlsx     - 特征重要性汇总")
        print("  18. shap_importance_summary.xlsx        - SHAP重要性汇总")
        print("  19. best_hyperparameters.csv            - 最佳超参数配置")
        print("  20. shap_r2_analysis.xlsx               - ★ SHAP-R² 详细分析（新增）")
        print("  21. shap_r2_summary_top20.csv           - ★ SHAP-R² Top20汇总（新增）")
        print("  22. shap_r2_cross_model_comparison.csv  - ★ 跨模型对比矩阵（新增）")
        
        print("\n【模型文件】")
        for model_name in trained_models.keys():
            print(f"  -  model_{model_name.replace(' ', '_')}.pkl")
        
        print("\n" + "=" * 100)
        print("Thanks for using the Metro particulate matter concentration prediction analysis system")
        print("=" * 100)
        print("\n")
        
        return results, trained_models, importance_results, shap_results, shap_r2_all
        
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到数据文件 '{file_path}'")
        print("请确保数据文件在当前目录下，或提供正确的文件路径")
        return None, None, None, None, None
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


# ==================== 15. 执行主函数====================
if __name__ == "__main__":
    # 修改为你的实际文件路径
    data_file = "file address."
    
    # 执行分析
    results, trained_models, importance_results, shap_results, shap_r2_all = main(data_file)
    
    # 如果需要，可以进一步分析结果
    if results is not None:
        print("\n" + "=" * 100)
        print("快速查看最佳模型")
        print("=" * 100)
        
        # 找出最佳模型
        best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
        best_results = results[best_model_name]
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"  测试集 R²:    {best_results['test_r2']:.4f}")
        print(f"  测试集 RMSE:  {best_results['test_rmse']:.4f}")
        print(f"  测试集 MAE:   {best_results['test_mae']:.4f}")
        print(f"  测试集 MAPE:  {best_results['test_mape']:.2f}%")
        print(f"  交叉验证 R²:  {best_results['cv_r2_mean']:.4f} ± {best_results['cv_r2_std']:.4f}")
        print(f"  过拟合差距:   {best_results['overfit_gap']:+.4f}")
        
        print("\n最佳超参数:")
        for param, value in best_results['best_params'].items():
            print(f"  {param}: {value}")
        
        # 显示Top 5特征
        if importance_results and best_model_name in importance_results and importance_results[best_model_name] is not None:
            print(f"\nTop 5 重要特征 ({best_model_name}):")
            top5 = importance_results[best_model_name].head(5)
            for idx, row in top5.items():
                print(f"  {idx}: {row:.6f}")
        
        # 显示Top 5 SHAP特征
        if shap_results and best_model_name in shap_results:
            if 'importance' in shap_results[best_model_name]:
                print(f"\nTop 5 SHAP重要特征 ({best_model_name}):")
                top5_shap = shap_results[best_model_name]['importance'].head(5)
                for idx, row in top5_shap.iterrows():
                    print(f"  {row['Feature']}: {row['Mean_Abs_SHAP']:.6f} {row['Direction']}")
        
        if shap_r2_all and best_model_name in shap_r2_all:
            print(f"\n★ Top 5 SHAP-based R² 贡献 ({best_model_name}):")
            print(f"  {'特征':<30} {'SHAP-R²':>10} {'占R²%':>10} {'方向':<12}")
            print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12}")
            top5_r2 = shap_r2_all[best_model_name].head(5)
            for _, row in top5_r2.iterrows():
                print(f"  {row['Feature']:<30} {row['SHAP_R2']:>10.5f} "
                      f"{row['SHAP_R2_Pct']:>9.2f}% {row['Direction']:<12}")
            
            top5_sum = top5_r2['SHAP_R2_Pct'].sum()
            print(f"\n  Top-5 累计贡献: {top5_sum:.2f}% (占模型R²的 {top5_sum:.2f}%)")
            print("\n" + "=" * 100)
            print("性能优化总结")
            print("=" * 100)
        
        
        # 显示所有模型性能
        print("\n所有模型性能排名:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        print(f"\n{'排名':<6} {'模型':<22} {'测试R²':<10} {'RMSE':<10} {'MAE':<10} {'过拟合':<10}")
        print("-" * 80)
        for rank, (model_name, model_results) in enumerate(sorted_models, 1):
            gap = model_results['overfit_gap']
            gap_status = "🟢" if gap <= 0.05 else "🟡" if gap <= 0.12 else "🟠" if gap <= 0.20 else "🔴"
            print(f"{rank:<6} {model_name:<22} {model_results['test_r2']:<10.4f} "
                  f"{model_results['test_rmse']:<10.4f} {model_results['test_mae']:<10.4f} "
                  f"{gap_status} {gap:+.4f}")
        
        print("\n" + "=" * 100)
        print("★ SHAP-based R² 分析亮点")
        print("=" * 100)
        
        if shap_r2_all and best_model_name in shap_r2_all:
            df_best_r2 = shap_r2_all[best_model_name]
            
            print(f"\n基于最优模型 {best_model_name} 的发现:")
            print(f"  • 模型总R²: {best_results['test_r2']:.4f}")
            print(f"  • 最关键特征: {df_best_r2.iloc[0]['Feature']}")
            print(f"    - 单独解释了 {df_best_r2.iloc[0]['SHAP_R2_Pct']:.2f}% 的R²")
            print(f"    - 影响方向: {df_best_r2.iloc[0]['Direction']}")
            
            # 帕累托分析
            cumsum_80_idx = (df_best_r2['Cumulative_Pct'] >= 80).idxmax()
            n_80 = cumsum_80_idx + 1 if cumsum_80_idx >= 0 else len(df_best_r2)
            print(f"\n  • 帕累托法则: 前 {n_80} 个特征解释了 80% 的R²")
            print(f"    占总特征数的 {n_80/len(df_best_r2)*100:.1f}%")
            
            # 正负影响统计
            pos_feats = df_best_r2[df_best_r2['Direction'] == '正影响(↑)']
            neg_feats = df_best_r2[df_best_r2['Direction'] == '负影响(↓)']
            pos_sum = pos_feats['SHAP_R2_Pct'].sum()
            neg_sum = neg_feats['SHAP_R2_Pct'].sum()
            
            print(f"\n  • 正负影响平衡:")
            print(f"    - 正影响特征（增大PM）: {len(pos_feats)} 个，总贡献 {pos_sum:.2f}%")
            print(f"    - 负影响特征（减小PM）: {len(neg_feats)} 个，总贡献 {neg_sum:.2f}%")
            
            if pos_sum > neg_sum * 1.5:
                print(f" 正影响主导，建议重点控制以下高贡献因素:")
                for _, row in pos_feats.head(3).iterrows():
                    print(f"       • {row['Feature']}: {row['SHAP_R2_Pct']:.2f}%")
            elif neg_sum > pos_sum * 1.5:
                print(f" 负影响主导，以下因素有效抑制PM:")
                for _, row in neg_feats.head(3).iterrows():
                    print(f" {row['Feature']}: {row['SHAP_R2_Pct']:.2f}%")
            else:
                print(f"    ○ 正负影响相对平衡")