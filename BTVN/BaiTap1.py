import numpy as np
import pandas as pd


# ==================================================
# 1. DỮ LIỆU RAM
# ==================================================

data = {
    "Dung_Luong_GB": [8, 8, 16, 16, 32,
                      8, 16, 16, 16, 32,
                      32, 4, 8, 4, 64],

    "Bus_MHz": [2666, 3200, 3200, 3200, 3200,
                4800, 4800, 5600, 6000, 5600,
                6000, 1600, 1600, 2400, 6000],

    "Gia_VND": [450000, 550000, 950000, 900000, 1850000,
                750000, 1350000, 1500000, 1800000, 2800000,
                3400000, 150000, 280000, 300000, 6500000]
}

df = pd.DataFrame(data)

print("===== DU LIEU RAM =====")
print(df)


# ==================================================
# 2. LẤY X VÀ y
# ==================================================

X = df[["Dung_Luong_GB", "Bus_MHz"]].values
y = df["Gia_VND"].values


# ==================================================
# 3. CHIA TRAIN VÀ TEST
# ==================================================

np.random.seed(42)

# Tạo vị trí các dòng
indices = np.arange(len(X))

# Trộn vị trí
np.random.shuffle(indices)

# 80% Train
train_size = int(0.8 * len(X))

train_index = indices[:train_size]
test_index = indices[train_size:]

X_train = X[train_index]
y_train = y[train_index]

X_test = X[test_index]
y_test = y[test_index]


print("\n===== CHIA DU LIEU =====")
print("So dong Train:", len(X_train))
print("So dong Test :", len(X_test))


# ==================================================
# 4. TẠO ĐẶC TRƯNG ĐA THỨC
# ==================================================

def tao_dac_trung(X, degree):

    dung_luong = X[:, 0]
    bus = X[:, 1]

    features = []

    for i in range(degree + 1):

        for j in range(degree + 1 - i):

            features.append(
                (dung_luong ** i) * (bus ** j)
            )

    return np.column_stack(features)


# ==================================================
# 5. HUẤN LUYỆN LINEAR REGRESSION
# ==================================================

def train_model(X, y, degree):

    X_poly = tao_dac_trung(X, degree)

    # Công thức Linear Regression
    w = np.linalg.pinv(
        X_poly.T @ X_poly
    ) @ X_poly.T @ y

    return w


# ==================================================
# 6. DỰ ĐOÁN
# ==================================================

def predict(X, w, degree):

    X_poly = tao_dac_trung(X, degree)

    return X_poly @ w


# ==================================================
# 7. TÍNH MAE
# ==================================================

def MAE(y_that, y_du_doan):

    return np.mean(
        np.abs(y_that - y_du_doan)
    )


# ==================================================
# 8. TẠO MODEL BỊ OVERFITTING
# ==================================================

degree_overfit = 8

w_overfit = train_model(
    X_train,
    y_train,
    degree_overfit
)

train_pred = predict(
    X_train,
    w_overfit,
    degree_overfit
)

test_pred = predict(
    X_test,
    w_overfit,
    degree_overfit
)

train_mae = MAE(
    y_train,
    train_pred
)

test_mae = MAE(
    y_test,
    test_pred
)


print("\n===== MODEL OVERFITTING =====")

print("Degree:", degree_overfit)

print("Train MAE:", train_mae)

print("Test MAE :", test_mae)


# ==================================================
# 9. K-FOLD CROSS VALIDATION
# ==================================================

def k_fold(X, y, k, degree):

    indices = np.arange(len(X))

    np.random.seed(42)

    np.random.shuffle(indices)

    folds = np.array_split(indices, k)

    errors = []


    for i in range(k):

        # Fold hiện tại làm Validation
        val_index = folds[i]

        # Các Fold còn lại làm Train
        train_index = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )


        X_train_fold = X[train_index]
        y_train_fold = y[train_index]

        X_val_fold = X[val_index]
        y_val_fold = y[val_index]


        # Train
        w = train_model(
            X_train_fold,
            y_train_fold,
            degree
        )


        # Dự đoán Validation
        y_val_pred = predict(
            X_val_fold,
            w,
            degree
        )


        # Tính MAE
        error = MAE(
            y_val_fold,
            y_val_pred
        )

        errors.append(error)


    # MAE trung bình của K Fold
    return np.mean(errors)


# ==================================================
# 10. THỬ CÁC DEGREE
# ==================================================

print("\n===== K-FOLD CROSS VALIDATION =====")

k = 5

results = {}

for degree in [1, 2, 3, 4, 5, 6, 7, 8]:

    cv_mae = k_fold(
        X_train,
        y_train,
        k,
        degree
    )

    results[degree] = cv_mae

    print(
        "Degree",
        degree,
        "-> CV MAE:",
        cv_mae
    )


# ==================================================
# 11. CHỌN DEGREE TỐT NHẤT
# ==================================================

best_degree = min(
    results,
    key=results.get
)


print("\n===== KET QUA K-FOLD =====")

print(
    "Degree tot nhat:",
    best_degree
)

print(
    "CV MAE:",
    results[best_degree]
)


# ==================================================
# 12. HUẤN LUYỆN MODEL CUỐI
# ==================================================

w_best = train_model(
    X_train,
    y_train,
    best_degree
)


train_pred = predict(
    X_train,
    w_best,
    best_degree
)

test_pred = predict(
    X_test,
    w_best,
    best_degree
)


train_mae = MAE(
    y_train,
    train_pred
)

test_mae = MAE(
    y_test,
    test_pred
)


print("\n===== MODEL SAU KHI K-FOLD =====")

print("Degree:", best_degree)

print("Train MAE:", train_mae)

print("Test MAE :", test_mae)


# ==================================================
# 13. NHẬP RAM MỚI ĐỂ DỰ ĐOÁN
# ==================================================

print("\n===== DU DOAN RAM MOI =====")

dung_luong = float(
    input("Nhap dung luong RAM (GB): ")
)

bus = float(
    input("Nhap Bus RAM (MHz): ")
)


ram_moi = np.array([
    [dung_luong, bus]
])


gia_du_doan = predict(
    ram_moi,
    w_best,
    best_degree
)


print("\n===== KET QUA =====")

print(
    "Dung luong:",
    dung_luong,
    "GB"
)

print(
    "Bus:",
    bus,
    "MHz"
)

print(
    "Gia du doan:",
    gia_du_doan[0],
    "VND"
)