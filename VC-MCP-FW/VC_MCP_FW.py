# تركيب المكتبات المطلوبة
!pip install transformers pandas numpy matplotlib scikit-learn torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import warnings
warnings.filterwarnings('ignore')

# البيانات المضمنة مباشرة في الكود
data = pd.DataFrame({
    'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
             2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
             2020, 2021, 2022, 2023, 2024],
    'Total Power Generation (Million MWH)': [29.025, 31.061, 32.731, 33.733, 30.938, 
                                             29.882, 31.538, 29.680, 28.993, 26.512,
                                             24.706, 24.108, 23.647, 24.881, 29.690,
                                             37.029, 42.946, 45.236, 40.749, 40.773,
                                             46.716, 47.323, 47.566, 42.684, 41.969]
})

print("=" * 60)
print("مكعب القوة الحديثة: تطبيق إطار العمل على بيانات الكهرباء")
print("=" * 60)
print("\nالبيانات المستخدمة في التحليل:")
print(data.head())

# المكون 1: Understand - فهم البيانات
print("\n" + "=" * 40)
print("المكون 1: فهم البيانات (Understand)")
print("=" * 40)

plt.figure(figsize=(14, 6))
plt.plot(data['Year'], data['Total Power Generation (Million MWH)'], 'bo-', linewidth=2, markersize=6)
plt.title('توليد الكهرباء على مر السنين', fontsize=14, fontweight='bold')
plt.xlabel('السنة', fontsize=12)
plt.ylabel('إجمالي توليد الكهرباء (مليون ميجاوات ساعة)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(data['Year'][::2], rotation=45)
plt.tight_layout()
plt.show()

# تحضير البيانات للتدريب
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data['Total Power Generation (Million MWH)'].values.reshape(-1, 1))

# تقسيم البيانات إلى تدريب واختبار - استخدام 70% للتدريب بدلاً من 80%
train_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# إنشاء متواليات زمنية للتدريب
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# تحويل البيانات إلى tensors مع الأبعاد الصحيحة لـ Transformer
X_train = torch.FloatTensor(X_train).transpose(1, 0)  # (seq_len, batch, features)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).transpose(1, 0) if len(X_test) > 0 else torch.FloatTensor()
y_test = torch.FloatTensor(y_test) if len(y_test) > 0 else torch.FloatTensor()

# تعريف نموذج Transformer مبسط للسلاسل الزمنية
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # طبقة الـ Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # إسقاط المدخلات إلى الفضاء المخفي
        x = self.input_projection(x)  # (seq_len, batch, hidden_size)
        
        # تمرير البيانات عبر محول Transformer
        x = self.transformer_encoder(x)  # (seq_len, batch, hidden_size)
        
        # أخذ آخر عنصر في التسلسل للتنبؤ
        x = x[-1, :, :]  # (batch, hidden_size)
        
        # إسقاط إلى حجم المخرجات
        x = self.output_projection(x)  # (batch, output_size)
        
        return x

# تهيئة النموذج
input_size = 1
hidden_size = 32
num_layers = 2
num_heads = 2
output_size = 1

model = TimeSeriesTransformer(input_size, hidden_size, num_layers, num_heads, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nجاري تدريب نموذج Transformer...")

# تدريب النموذج
num_epochs = 300
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# المكون 2: Conclude - استخلاص النتائج
print("\n" + "=" * 40)
print("المكون 2: استخلاص النتائج (Conclude)")
print("=" * 40)

# تقييم النموذج
model.eval()
with torch.no_grad():
    train_predictions = model(X_train)
    
    # عكس التحجيم للحصول على القيم الحقيقية
    train_predictions = scaler.inverse_transform(train_predictions.numpy().reshape(-1, 1))
    y_train_actual = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
    
    # معالجة بيانات الاختبار إذا كانت موجودة
    if X_test.size(0) > 0:
        test_predictions = model(X_test)
        test_predictions = scaler.inverse_transform(test_predictions.numpy().reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    else:
        test_predictions = np.array([])
        y_test_actual = np.array([])

# حساب دقة النموذج
def calculate_mae(actual, predicted):
    if len(actual) == 0:
        return 0
    return np.mean(np.abs(actual - predicted))

train_mae = calculate_mae(y_train_actual, train_predictions)
test_mae = calculate_mae(y_test_actual, test_predictions) if len(y_test_actual) > 0 else 0

print(f'متوسط الخطأ المطلق (MAE) على بيانات التدريب: {train_mae:.4f}')
if len(y_test_actual) > 0:
    print(f'متوسط الخطأ المطلق (MAE) على بيانات الاختبار: {test_mae:.4f}')
else:
    print('لا توجد بيانات اختبار كافية للتقييم')

# المكون 3: Apply - تطبيق النتائج
print("\n" + "=" * 40)
print("المكون 3: تطبيق النتائج (Apply)")
print("=" * 40)

# التنبؤ بالسنوات القادمة
future_years = 3
last_sequence = scaled_data[-seq_length:].reshape(seq_length, 1, 1)  # (seq_len, batch, features)
last_sequence = torch.FloatTensor(last_sequence)

future_predictions = []
current_sequence = last_sequence.clone()

for _ in range(future_years):
    with torch.no_grad():
        next_pred = model(current_sequence)
        future_predictions.append(next_pred.item())
        
        # تحديث المتوالية بإضافة التنبؤ الجديد وإزالة أقدم قيمة
        new_sequence = torch.cat([current_sequence[1:, :, :], 
                                 next_pred.reshape(1, 1, 1)], dim=0)
        current_sequence = new_sequence

# عكس التحجيم للتنبؤات المستقبلية
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# إنشاء سنوات مستقبلية
last_year = data['Year'].iloc[-1]
future_years_list = [last_year + i + 1 for i in range(future_years)]

print("التنبؤ بتوليد الكهرباء للسنوات القادمة باستخدام Transformer:")
for year, prediction in zip(future_years_list, future_predictions):
    print(f"{year}: {prediction[0]:.3f} مليون ميجاوات ساعة")

# المكون 4: Affect - قياس التأثير
print("\n" + "=" * 40)
print("المكون 4: قياس التأثير (Affect)")
print("=" * 40)

# تحليل أنماط الاستهلاك واقتراح استراتيجيات الترشيد
avg_consumption = np.mean(data['Total Power Generation (Million MWH)'])
max_consumption = np.max(data['Total Power Generation (Million MWH)'])
min_consumption = np.min(data['Total Power Generation (Million MWH)'])

print(f"متوسط استهلاك الكهرباء: {avg_consumption:.3f} مليون ميجاوات ساعة")
print(f"أعلى استهلاك: {max_consumption:.3f} مليون ميجاوات ساعة (سنة {data['Year'][data['Total Power Generation (Million MWH)'].idxmax()]})")
print(f"أقل استهلاك: {min_consumption:.3f} مليون ميجاوات ساعة (سنة {data['Year'][data['Total Power Generation (Million MWH)'].idxmin()]})")

# تحديد فترات الذروة
peak_threshold = avg_consumption * 1.1
peak_years = data[data['Total Power Generation (Million MWH)'] > peak_threshold]
print(f"\nسنوات ذروة الاستهلاك (أعلى من {peak_threshold:.2f}):")
print(peak_years[['Year', 'Total Power Generation (Million MWH)']].to_string(index=False))

# اقتراح استراتيجيات الترشيد بناءً على تحليل المحولات
print("\n" + "=" * 50)
print("استراتيجيات مقترحة لترشيد استهلاك الكهرباء (بناءً على تحليل Transformer):")
print("=" * 50)
print("1. استخدام أنظمة الذكاء الاصطناعي القائمة على المحولات للتنبؤ بدقة بالطلب على الطاقة")
print("2. تحسين كفاءة محطات التوليد خلال فترات الذروة باستخدام نماذج الانتباه (Attention)")
print("3. تطوير أنظمة تحكم ذكية تستخدم محولات Transformer لإدارة الأحمال بشكل ديناميكي")
print("4. الاستثمار في تقنيات الطاقة المتجددة مع استخدام نماذج المحولات للتنبؤ بالتوليد")
print("5. تطبيق برامج توعوية مبنية على تحليلات المحولات لفهم أنماط الاستهلاك")
print("6. استخدام أنظمة التخزين بالبطاريات مع نماذج التنبؤ القائمة على المحولات")

# تصور النتائج النهائية
plt.figure(figsize=(16, 10))

# الرسم 1: البيانات التاريخية والتنبؤات
plt.subplot(2, 2, 1)
plt.plot(data['Year'], data['Total Power Generation (Million MWH)'], 'bo-', 
         linewidth=2, markersize=6, label='البيانات التاريخية')
plt.plot(future_years_list, future_predictions, 'ro--', 
         linewidth=2, markersize=8, label='التنبؤات المستقبلية (Transformer)')
plt.title('توليد الكهرباء والتنبؤ بالمستقبل باستخدام المحولات', fontsize=14, fontweight='bold')
plt.xlabel('السنة', fontsize=12)
plt.ylabel('إجمالي توليد الكهرباء (مليون ميجاوات ساعة)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# الرسم 2: فقدان التدريب
plt.subplot(2, 2, 2)
plt.plot(losses, 'r-', linewidth=2)
plt.title('فقدان التدريب خلال التكرارات (نموذج Transformer)', fontsize=14, fontweight='bold')
plt.xlabel('عدد التكرارات', fontsize=12)
plt.ylabel('قيمة الفقدان', fontsize=12)
plt.grid(True, alpha=0.3)

# الرسم 3: مقارنة التنبؤات مع القيم الفعلية
plt.subplot(2, 2, 3)
plt.scatter(y_train_actual, train_predictions, alpha=0.7, label='بيانات التدريب')
if len(y_test_actual) > 0:
    plt.scatter(y_test_actual, test_predictions, alpha=0.7, label='بيانات الاختبار', color='red')
min_val = min(np.min(y_train_actual), np.min(train_predictions))
max_val = max(np.max(y_train_actual), np.max(train_predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
plt.title('مقارنة التنبؤات مع القيم الفعلية (Transformer)', fontsize=14, fontweight='bold')
plt.xlabel('القيم الفعلية', fontsize=12)
plt.ylabel('التنبؤات', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# الرسم 4: استهلاك الكهرباء خلال السنوات
plt.subplot(2, 2, 4)
plt.bar(data['Year'], data['Total Power Generation (Million MWH)'], 
        alpha=0.7, color='green')
plt.axhline(y=avg_consumption, color='r', linestyle='--', linewidth=2, 
            label=f'المتوسط: {avg_consumption:.2f}')
plt.title('استهلاك الكهرباء خلال السنوات', fontsize=14, fontweight='bold')
plt.xlabel('السنة', fontsize=12)
plt.ylabel('إجمالي توليد الكهرباء', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(data['Year'][::2], rotation=45)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("تم الانتهاء من تطبيق إطار عمل 'مكعب القوة الحديثة' بنجاح باستخدام المحولات!")
print("=" * 60)