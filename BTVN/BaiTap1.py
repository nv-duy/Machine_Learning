import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error


# ==========================================
# 1. TẠO DATASET
# ==========================================

data = {
    'Dung_Luong_GB': [
        8, 8, 16, 16, 32,
        8, 16, 16, 16, 32,
        32, 4, 8, 4, 64
    ],

    'Bus_MHz': [
        2666, 3200, 3200, 3200, 3200,
        4800, 4800, 5600, 6000, 5600,
        6000, 1600, 1600, 2400, 6000
    ],

    'Gia_VND': [
        450000, 550000, 950000, 900000, 1850000,
        750000, 1350000, 1500000, 1800000, 2800000,
        3400000, 150000, 280000, 300000, 6500000
    ]
}

df = pd.DataFrame(data)


# ==========================================
# 2. TÁCH X VÀ y
# ==========================================

X = df[['Dung_Luong_GB', 'Bus_MHz']]

y = df['Gia_VND']


# ==========================================
# 3. CHIA TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. TẠO MODEL OVERFITTING
# ==========================================

model_overfit = DecisionTreeRegressor(
    random_state=42
)

model_overfit.fit(X_train, y_train)


# ==========================================
# 5. KIỂM TRA OVERFITTING
# ==========================================

train_pred = model_overfit.predict(X_train)
test_pred = model_overfit.predict(X_test)

train_mae = mean_absolute_error(
    y_train,
    train_pred
)

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

print("========== TRUOC K-FOLD ==========")

print("Train MAE:", train_mae)
print("Test MAE :", test_mae)


# ==========================================
# 6. K-FOLD
# ==========================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 7. THỬ NHIỀU MAX_DEPTH
# ==========================================

results = {}

for depth in [1, 2, 3, 4, 5]:

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring='neg_mean_absolute_error'
    )

    mae = -scores.mean()

    results[depth] = mae

    print(
        "max_depth =",
        depth,
        "| CV MAE =",
        mae
    )


# ==========================================
# 8. CHỌN MAX_DEPTH TỐT NHẤT
# ==========================================

best_depth = min(
    results,
    key=results.get
)

print("\nBest max_depth:", best_depth)
print("Best CV MAE:", results[best_depth])


# ==========================================
# 9. TRAIN MODEL MỚI
# ==========================================

best_model = DecisionTreeRegressor(
    max_depth=best_depth,
    random_state=42
)

best_model.fit(X_train, y_train)


# ==========================================
# 10. ĐÁNH GIÁ SAU K-FOLD
# ==========================================

train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

train_mae = mean_absolute_error(
    y_train,
    train_pred
)

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

print("\n========== SAU K-FOLD ==========")

print("Train MAE:", train_mae)
print("Test MAE :", test_mae)


# ==========================================
# 11. nHẬP 1 LOẠI RAM MỚI VÀ DỰ ĐOÁN
# ==========================================

dung_luong = float(input("Nhap dung luong RAM (GB): "))
bus = float(input("Nhap Bus RAM (MHz): "))

ram_moi = [[dung_luong, bus]]

gia_du_doan = best_model.predict(ram_moi)

print("\n========== KET QUA ==========")
print("Dung luong:", dung_luong, "GB")
print("Bus:", bus, "MHz")
print("Gia du doan:", gia_du_doan[0], "VND")