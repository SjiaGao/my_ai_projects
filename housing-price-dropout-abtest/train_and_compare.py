import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 设置随机种子
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# 1. 数据加载 + 特征工程（完全一致）
# ----------------------------
data = pd.read_csv('housing.csv', header=None, sep=r'\s+')
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
data.columns = feature_names + ['PRICE']

# 特征工程（与你优化版完全一致）
features = data.copy()
features['RM_LSTAT'] = features['RM'] * features['LSTAT']
features['INV_DIS'] = 1.0 / (features['DIS'] + 1e-6)
features['CRIM_NOX'] = features['CRIM'] * features['NOX']

x = features[feature_names + ['RM_LSTAT', 'INV_DIS', 'CRIM_NOX']].values
y = features['PRICE'].values.reshape(-1, 1)

# 标准化
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

# 划分数据
train_x, test_x, train_y, test_y = train_test_split(
    x_scaled, y_scaled, test_size=0.25, random_state=SEED
)
true_y = scaler_y.inverse_transform(test_y).flatten()


# ----------------------------
# 2. 模型构建函数（唯一区别：use_dropout）
# ----------------------------
def create_model(use_dropout=False, input_dim=x.shape[1]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(0.3))  # 尝试稍高一点（0.3）
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    return model


# ----------------------------
# 3. 训练函数（使用 Huber Loss）
# ----------------------------
def train_model(model, name, train_x, train_y, test_x, scaler_y, true_y):
    print(f"\n🚀 训练: {name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='huber_loss',
        metrics=['mae']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=25, restore_best_weights=True
    )

    history = model.fit(
        train_x, train_y,
        epochs=300,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    # 预测 & 还原
    pred_y_scaled = model.predict(test_x, verbose=0)
    pred_y = scaler_y.inverse_transform(pred_y_scaled).flatten()

    # 指标
    mae = mean_absolute_error(true_y, pred_y)
    r2 = r2_score(true_y, pred_y)

    # 高价房 MAE
    high_mask = true_y > 35
    high_mae = mean_absolute_error(true_y[high_mask], pred_y[high_mask]) if np.any(high_mask) else np.nan

    return {
        'name': name,
        'pred_y': pred_y,
        'mae': mae,
        'r2': r2,
        'high_mae': high_mae,
        'history': history
    }


# ----------------------------
# 4. 运行 A/B 实验
# ----------------------------
results = []

# A: No Dropout
model_a = create_model(use_dropout=False)
res_a = train_model(model_a, "No Dropout", train_x, train_y, test_x, scaler_y, true_y)
results.append(res_a)

# B: With Dropout
model_b = create_model(use_dropout=True)
res_b = train_model(model_b, "With Dropout", train_x, train_y, test_x, scaler_y, true_y)
results.append(res_b)

# ----------------------------
# 5. 打印对比结果
# ----------------------------
print("\n" + "=" * 70)
print("📊 Dropout A/B Test（基于优化特征 + Huber Loss）")
print("=" * 70)
print(f"{'模型':<15} {'MAE':<8} {'R²':<8} {'High-Price MAE (>35k)':<20}")
print("-" * 70)
for res in results:
    high_mae_str = f"{res['high_mae']:.2f}" if not np.isnan(res['high_mae']) else "N/A"
    print(f"{res['name']:<15} {res['mae']:<8.2f} {res['r2']:<8.3f} {high_mae_str:<20}")
print("=" * 70)

# ----------------------------
# 6. 可视化对比
# ----------------------------
plt.figure(figsize=(15, 4))

# 子图1: 验证 loss 对比
plt.subplot(1, 3, 1)
for res in results:
    plt.plot(res['history'].history['val_loss'], label=res['name'])
plt.title('Validation Loss (Huber)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 子图2: 预测 vs 真实（No Dropout）
plt.subplot(1, 3, 2)
plt.scatter(true_y, results[0]['pred_y'], alpha=0.7, label='Predict')
plt.plot([true_y.min(), true_y.max()], [true_y.min(), true_y.max()], 'r--')
plt.title(results[0]['name'])
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.grid(True)

# 子图3: 预测 vs 真实（With Dropout）
plt.subplot(1, 3, 3)
plt.scatter(true_y, results[1]['pred_y'], alpha=0.7, color='orange', label='Predict')
plt.plot([true_y.min(), true_y.max()], [true_y.min(), true_y.max()], 'r--')
plt.title(results[1]['name'])
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------------
# 7. 残差对比（可选）
# ----------------------------
plt.figure(figsize=(12, 4))
for i, res in enumerate(results):
    residuals = res['pred_y'] - true_y
    plt.subplot(1, 2, i + 1)
    plt.scatter(true_y, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residuals: {res['name']}")
    plt.xlabel('True Price')
    plt.ylabel('Error')
    plt.grid(True)
plt.tight_layout()
plt.show()