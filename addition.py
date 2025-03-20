import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
class SimpleAdditionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,2)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

class BetterAdditionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入特征数为6: a_int_norm, a_dec_scaled, b_int_norm, b_dec_scaled, c_int_norm, c_dec_scaled
        self.fc1 = nn.Linear(6, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        # 直接从输入到最后一层的残差连接
        self.skip = nn.Linear(6, 64)
        
        self.fc_out = nn.Linear(64, 2)
    
    def forward(self, x):
        # 保存原始输入用于残差连接
        identity = x
        
        # 主路径
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # 残差连接
        identity = self.skip(identity)
        x = x + identity
        
        # 输出层
        x = torch.softmax(self.fc_out(x), dim=1)
        return x

def normalize_data_log(a, b, c, max_value=10000):
    """将数值分解为整数部分和小数部分，根据数量级设置不同精度"""
    # 确定小数保留位数
    if max_value <= 10:
        decimal_places = 1
    elif max_value <= 100:
        decimal_places = 2
    elif max_value <= 1000:
        decimal_places = 3
    else:
        decimal_places = 4
    
    # 截断到指定小数位
    factor = 10**decimal_places
    a_truncated = round(a * factor) / factor
    b_truncated = round(b * factor) / factor
    c_truncated = round(c * factor) / factor
    
    # 分离整数和小数部分
    a_int, a_dec = int(a_truncated), a_truncated - int(a_truncated)
    b_int, b_dec = int(b_truncated), b_truncated - int(b_truncated)
    c_int, c_dec = int(c_truncated), c_truncated - int(c_truncated)
    
    # 使用log1p归一化整数部分
    max_log = np.log1p(max_value * 2)
    a_int_norm = np.log1p(a_int) / max_log
    b_int_norm = np.log1p(b_int) / max_log
    c_int_norm = np.log1p(c_int) / max_log
    
    # 小数部分只保留指定精度的位数，并放大
    decimal_scale = 10**decimal_places
    a_dec_scaled = int(a_dec * decimal_scale)
    b_dec_scaled = int(b_dec * decimal_scale)
    c_dec_scaled = int(c_dec * decimal_scale)
    
    # 再使用log1p归一化小数部分
    max_dec_log = np.log1p(decimal_scale)
    a_dec_norm = np.log1p(a_dec_scaled) / max_dec_log
    b_dec_norm = np.log1p(b_dec_scaled) / max_dec_log
    c_dec_norm = np.log1p(c_dec_scaled) / max_dec_log
    
    return [a_int_norm, a_dec_norm, b_int_norm, b_dec_norm, c_int_norm, c_dec_norm]

def normalize_data_simple(a, b, c, max_value=10000):
    """使用简单线性归一化，更适合加法任务"""
    # 计算适当的小数位数
    if max_value <= 10:
        decimal_places = 1
    elif max_value <= 100:
        decimal_places = 2
    elif max_value <= 1000:
        decimal_places = 3
    else:
        decimal_places = 4
    
    # 截断到指定小数位
    factor = 10**decimal_places
    a_truncated = round(a * factor) / factor
    b_truncated = round(b * factor) / factor
    c_truncated = round(c * factor) / factor
    
    # 分离整数和小数部分
    a_int, a_dec = int(a_truncated), a_truncated - int(a_truncated)
    b_int, b_dec = int(b_truncated), b_truncated - int(b_truncated)
    c_int, c_dec = int(c_truncated), c_truncated - int(c_truncated)
    
    # 线性归一化
    a_int_norm = a_int / max_value
    b_int_norm = b_int / max_value
    c_int_norm = c_int / max_value
    
    return [a_int_norm, a_dec, b_int_norm, b_dec, c_int_norm, c_dec]

def normalize_data_better(a, b, c, max_value=10000):
    """更好的归一化方法：直接归一化整个数值并添加差值特征"""
    # 直接归一化整个数值
    a_norm = a / max_value
    b_norm = b / max_value
    c_norm = c / max_value
    
    # 添加a+b和c之间的差值特征
    sum_diff = abs((a + b) - c) / max_value
    
    # 添加a+b是否约等于c的特征(0或1)
    # 根据数值大小确定精度
    if max(a, b, c) <= 10:
        precision = 1e-1
    elif max(a, b, c) <= 100:
        precision = 1e-2
    elif max(a, b, c) <= 1000:
        precision = 1e-3
    else:
        precision = 1e-4
    
    is_equal_feature = 1.0 if abs((a + b) - c) < precision else 0.0
    
    return [a_norm, b_norm, c_norm, sum_diff, is_equal_feature]

def normalize_data_user(a, b, c, max_value=10000, is_correct_addition=None):
    """按照用户要求的归一化方法：分离整数和小数部分，并根据精度放大小数部分
    
    参数:
        a, b, c: 输入的数值
        max_value: 最大值范围
        is_correct_addition: 是否为正确加法，如果为None，则自动检测
    """
    # 确定小数保留位数
    if max_value <= 10:
        decimal_places = 1
        scale_factor = 10
    elif max_value <= 100:
        decimal_places = 2
        scale_factor = 100
    elif max_value <= 1000:
        decimal_places = 3
        scale_factor = 1000
    else:
        decimal_places = 4
        scale_factor = 10000
    
    # 分离整数和小数部分
    a_int, a_dec = int(a), a - int(a)
    b_int, b_dec = int(b), b - int(b)
    c_int, c_dec = int(c), c - int(c)
    
    # 放大小数部分并取整
    a_dec_int = int(a_dec * scale_factor)
    b_dec_int = int(b_dec * scale_factor)
    
    # 如果未指定是否正确加法，自动检测
    if is_correct_addition is None:
        # 使用较小的容差，更精确地检测
        factor = 10**decimal_places
        a_truncated = round(a * factor) / factor
        b_truncated = round(b * factor) / factor
        c_truncated = round(c * factor) / factor
        is_correct_addition = abs((a_truncated + b_truncated) - c_truncated) < (0.5 * 10**(-decimal_places))
    
    # 对于正确加法，使用小数部分之和的模
    if is_correct_addition:
        c_dec_int = (a_dec_int + b_dec_int) % scale_factor
    else:
        # 对于错误加法，保留原始c的小数部分
        c_dec_int = int(c_dec * scale_factor)
    
    # 归一化整数部分 - 使用2*max_value确保c不会超过1
    a_int_norm = a_int / (2 * max_value)
    b_int_norm = b_int / (2 * max_value)
    c_int_norm = c_int / (2 * max_value)
    
    # 归一化放大后的小数整数部分
    a_dec_scaled = a_dec_int / scale_factor
    b_dec_scaled = b_dec_int / scale_factor
    c_dec_scaled = c_dec_int / scale_factor
    
    return [a_int_norm, a_dec_scaled, b_int_norm, b_dec_scaled, c_int_norm, c_dec_scaled]

def generate_data(num_samples=10000, max_value=10000, decimal_precision=4):
    """根据指定的小数精度生成数据"""
    data = []
    labels = []
    original_values = []  # 存储原始的a,b,c值
    
    # 样本细分
    regular_samples = int(num_samples * 0.5)  # 普通样本
    edge_samples = int(num_samples * 0.3)     # 边界样本
    hard_samples = num_samples - regular_samples - edge_samples  # 困难样本
    
    # 确定截断因子
    factor = 10**decimal_precision
    
    # 1. 生成普通样本
    for _ in range(regular_samples):
        a = np.random.uniform(0, max_value)
        b = np.random.uniform(0, max_value)
        
        # 截断为指定小数位数
        a_truncated = round(a * factor) / factor
        b_truncated = round(b * factor) / factor
        exact_sum = a_truncated + b_truncated
        
        if np.random.random() < 0.5:
            c_truncated = exact_sum  # 正确答案
            label = [0, 1]
            is_correct = True
        else:
            # 错误答案，确保错误发生在指定小数位之内
            error_magnitude = 10**(-decimal_precision) * np.random.uniform(1, 9)
            c_truncated = exact_sum + error_magnitude * (1 if np.random.random() < 0.5 else -1)
            label = [1, 0]
            is_correct = False
        
        # 使用截断后的值生成特征
        features = normalize_data_user(a_truncated, b_truncated, c_truncated, max_value, is_correct_addition=is_correct)
        data.append(features)
        labels.append(label)
        original_values.append((a_truncated, b_truncated, c_truncated))  # 保存截断后的值
    
    # 2. 生成边界样本 - 非常接近正确答案的错误样本
    for _ in range(edge_samples):
        a = np.random.uniform(0, max_value)
        b = np.random.uniform(0, max_value)
        
        # 截断为指定小数位数
        a_truncated = round(a * factor) / factor
        b_truncated = round(b * factor) / factor
        exact_sum = a_truncated + b_truncated
        
        # 有一半概率生成边界正确样本
        if np.random.random() < 0.5:
            c_truncated = exact_sum  # 正确答案
            label = [0, 1]
            is_correct = True
        else:
            # 非常接近但不等于正确答案的样本
            tiny_error = 10**(-decimal_precision-1) * np.random.uniform(1, 9)
            c_truncated = exact_sum + tiny_error
            label = [1, 0]
            is_correct = False
        
        # 使用截断后的值生成特征
        features = normalize_data_user(a_truncated, b_truncated, c_truncated, max_value, is_correct_addition=is_correct)
        data.append(features)
        labels.append(label)
        original_values.append((a_truncated, b_truncated, c_truncated))
    
    # 3. 生成困难样本
    for _ in range(hard_samples):
        sample_type = np.random.choice(['large_small', 'nearly_equal', 'exact_decimal'])
        
        if sample_type == 'large_small':
            # 一个很大一个很小的数
            a = np.random.uniform(max_value*0.7, max_value)
            b = np.random.uniform(0.00001, max_value*0.001)
            
            # 截断为指定小数位数
            a_truncated = round(a * factor) / factor
            b_truncated = round(b * factor) / factor
            exact_sum = a_truncated + b_truncated
            
            if np.random.random() < 0.5:
                # 正确答案
                c_truncated = exact_sum
                label = [0, 1]
                is_correct = True
            else:
                # 忽略小数部分导致的错误
                c_truncated = a_truncated  # 直接忽略b
                label = [1, 0]
                is_correct = False
                
        elif sample_type == 'nearly_equal':
            # 接近相等的两个数
            base = np.random.uniform(0, max_value/2)
            a = base + np.random.uniform(-0.01, 0.01) * base
            b = base + np.random.uniform(-0.01, 0.01) * base
            
            # 截断为指定小数位数
            a_truncated = round(a * factor) / factor
            b_truncated = round(b * factor) / factor
            exact_sum = a_truncated + b_truncated
            
            if np.random.random() < 0.5:
                # 正确答案
                c_truncated = exact_sum
                label = [0, 1]
                is_correct = True
            else:
                # 接近但不等于正确答案
                c_truncated = round(2 * base * factor) / factor  # 使用2*base作为近似值
                label = [1, 0]
                is_correct = False
                
        else:  # exact_decimal
            # 小数位精确相加
            decimal_places = decimal_precision
            a = np.random.uniform(0, max_value)
            b = (10**(-decimal_places)) * np.random.randint(1, 9)
            
            # 截断为指定小数位数
            a_truncated = round(a * factor) / factor
            b_truncated = round(b * factor) / factor
            exact_sum = a_truncated + b_truncated
            
            if np.random.random() < 0.5:
                # 正确答案
                c_truncated = exact_sum
                label = [0, 1]
                is_correct = True
            else:
                # 错误：忽略小数位
                c_truncated = a_truncated
                label = [1, 0]
                is_correct = False
        
        # 使用截断后的值生成特征
        features = normalize_data_user(a_truncated, b_truncated, c_truncated, max_value, is_correct_addition=is_correct)
        data.append(features)
        labels.append(label)
        original_values.append((a_truncated, b_truncated, c_truncated))
    
    # 打乱数据
    combined = list(zip(data, labels, original_values))
    np.random.shuffle(combined)
    data, labels, original_values = zip(*combined)
    
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), original_values

def generate_mixed_data(num_samples=80000):
    """生成混合难度的数据集，包含不同范围的加法问题和困难案例"""
    data = []
    labels = []
    original_values = []  # 存储原始的a,b,c值
    
    # 每个难度范围的样本数量
    samples_per_range = num_samples // 4  # 平均分配到4个难度级别
    
    # 生成四个不同难度的数据
    for max_value in [10, 100, 1000, 10000]:
        # 普通样本数量
        base_samples = int(samples_per_range * 0.7)
        hard_samples = samples_per_range - base_samples
        
        # 确定小数保留位数
        if max_value <= 10:
            decimal_places = 1
        elif max_value <= 100:
            decimal_places = 2
        elif max_value <= 1000:
            decimal_places = 3
        else:
            decimal_places = 4
            
        # 确定截断因子
        factor = 10**decimal_places
        
        # 普通样本
        for _ in range(base_samples):
            a = np.random.uniform(0, max_value)
            b = np.random.uniform(0, max_value)
            
            # A. 截断为指定小数位数
            a_truncated = round(a * factor) / factor
            b_truncated = round(b * factor) / factor
            exact_sum = a_truncated + b_truncated
            
            if np.random.random() < 0.5:
                c_truncated = exact_sum  # 正确答案
                label = [0, 1]
                is_correct = True
            else:
                error_rate = np.random.uniform(0.01, 0.1)
                error = exact_sum * error_rate * (1 if np.random.random() < 0.5 else -1)
                c_truncated = exact_sum + error
                label = [1, 0]
                is_correct = False
            
            # B. 使用当前难度级别的max_value和is_correct
            features = normalize_data_user(a_truncated, b_truncated, c_truncated, max_value, is_correct_addition=is_correct)
            data.append(features)
            labels.append(label)
            original_values.append((a_truncated, b_truncated, c_truncated))  # 保存截断后的值
        
        # 困难样本：一大一小
        n_hard1 = hard_samples // 3
        for _ in range(n_hard1):
            a = np.random.uniform(max_value*0.8, max_value)
            b = np.random.uniform(0.0001, 0.01)
            
            # A. 截断为指定小数位数
            a_truncated = round(a * factor) / factor
            b_truncated = round(b * factor) / factor
            exact_sum = a_truncated + b_truncated
            
            if np.random.random() < 0.5:
                c_truncated = exact_sum
                label = [0, 1]
                is_correct = True
            else:
                error = np.random.uniform(0.0001, 0.001) * (1 if np.random.random() < 0.5 else -1)
                c_truncated = exact_sum + error
                label = [1, 0]
                is_correct = False
            
            # B. 使用当前难度级别的max_value和is_correct
            features = normalize_data_user(a_truncated, b_truncated, c_truncated, max_value, is_correct_addition=is_correct)
            data.append(features)
            labels.append(label)
            original_values.append((a_truncated, b_truncated, c_truncated))  # 保存截断后的值
        
        # 困难样本：小数位精度测试
        n_hard2 = hard_samples - n_hard1
        for _ in range(n_hard2):
            a = np.random.uniform(0, max_value)
            b = np.random.uniform(0, max_value)
            
            # A. 截断为指定小数位数
            a_truncated = round(a * factor) / factor
            b_truncated = round(b * factor) / factor
            exact_sum = a_truncated + b_truncated
            
            if np.random.random() < 0.5:
                c_truncated = exact_sum
                label = [0, 1]
                is_correct = True
            else:
                # 非常接近但不等于正确答案
                tiny_error = np.random.uniform(0.00001, 0.0001) * (1 if np.random.random() < 0.5 else -1)
                c_truncated = exact_sum + tiny_error
                label = [1, 0]
                is_correct = False
                
            # B. 使用当前难度级别的max_value和is_correct
            features = normalize_data_user(a_truncated, b_truncated, c_truncated, max_value, is_correct_addition=is_correct)
            data.append(features)
            labels.append(label)
            original_values.append((a_truncated, b_truncated, c_truncated))  # 保存截断后的值
    
    # 打乱数据
    combined = list(zip(data, labels, original_values))
    np.random.shuffle(combined)
    data, labels, original_values = zip(*combined)
    
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), original_values

def check_addition(model, a, b, c, max_value=10000):
    model.eval()
    with torch.no_grad():
        # 确定小数保留位数
        if max_value <= 10:
            decimal_places = 1
        elif max_value <= 100:
            decimal_places = 2
        elif max_value <= 1000:
            decimal_places = 3
        else:
            decimal_places = 4
            
        # 按照指定小数位数进行加法验证
        factor = 10**decimal_places
        a_truncated = round(a * factor) / factor
        b_truncated = round(b * factor) / factor
        c_truncated = round(c * factor) / factor
        
        # 验证加法是否正确，考虑指定小数位的精度
        correct = abs((a_truncated + b_truncated) - c_truncated) < (0.5 * 10**(-decimal_places))
        
        # 使用模型预测 - 使用截断后的值生成特征
        features = normalize_data_user(a_truncated, b_truncated, c_truncated, max_value)
        input_data = torch.tensor([features], dtype=torch.float32).to(device)
        probabilities = model(input_data)
        
        return probabilities[0, 1].item()

def curriculum_training(model, loss_fn, optimizer, scheduler, epochs_per_stage=100, final_stage_epochs=300, stage_losss_break=0.3, final_accuracy_break=0.95, stage_transition_callback=None):
    # 修改阶段设置，更平滑的难度进阶
    stages = [
        {"max_value": 10, "name": "10以内加法", "decimal_precision": 1, "epochs": 30},
        {"max_value": 50, "name": "50以内加法", "decimal_precision": 1, "epochs": 30},
        {"max_value": 100, "name": "100以内加法", "decimal_precision": 2, "epochs": 70},
        {"max_value": 500, "name": "500以内加法", "decimal_precision": 2, "epochs": 70},
        {"max_value": 1000, "name": "1000以内加法", "decimal_precision": 3, "epochs": 100},
        {"max_value": 5000, "name": "5000以内加法", "decimal_precision": 3, "epochs": 100},
        {"max_value": 10000, "name": "10000以内加法", "decimal_precision": 4, "epochs": 120}
    ]
    
    # 保存阶段数据，用于混合训练
    stage_data = []
    
    # 为每个阶段保存一个检查点
    for stage_idx, stage in enumerate(stages):
        print(f"\n开始训练阶段 {stage_idx+1}: {stage['name']}")
        max_value = stage["max_value"]
        decimal_precision = stage["decimal_precision"]
        cur_epochs = stage["epochs"]  # 使用每个阶段自定义的轮次数
        
        # 为当前难度生成数据
        X_train, y_train, train_values = generate_data(num_samples=8000, max_value=max_value, decimal_precision=decimal_precision)
        X_val, y_val, val_values = generate_data(num_samples=1000, max_value=max_value, decimal_precision=decimal_precision)
        
        # 保存数据用于后续混合
        stage_data.append((X_train[:1000], y_train[:1000], train_values[:1000]))
        
        # 将数据移到设备
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        
        # 为高精度阶段添加热身，先用低精度的方式训练一下
        if stage_idx >= 2:  # 从第三阶段开始
            print(f"阶段{stage_idx+1}热身训练...")
            # 创建低精度版本的数据
            warmup_precision = max(1, decimal_precision - 1)
            X_warmup, y_warmup, warmup_values = generate_data(
                num_samples=3000, max_value=max_value, decimal_precision=warmup_precision
            )
            X_warmup = X_warmup.to(device)
            y_warmup = y_warmup.to(device)
            
            # 热身10轮
            for warmup_epoch in range(10):
                model.train()
                indices = torch.randperm(len(X_warmup))
                batch_size = 128
                total_loss = 0
                
                for i in range(0, len(X_warmup), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    inputs = X_warmup[batch_indices]
                    targets = y_warmup[batch_indices]
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                warmup_loss = total_loss / (len(X_warmup) / batch_size)
                print(f"热身 轮次 {warmup_epoch+1}/10, 损失: {warmup_loss:.4f}")
        
        # 训练当前阶段
        batch_size = 128
        best_accuracy = 0
        for epoch in range(cur_epochs):
            # 训练模式
            model.train()
            total_loss = 0
            indices = torch.randperm(len(X_train))
            
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                inputs = X_train[batch_indices]
                targets = y_train[batch_indices]
                
                # 前向传播
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            train_loss = total_loss / (len(X_train) / batch_size)
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = loss_fn(val_outputs, y_val).item()
                
                # 计算准确率 - 使用当前阶段的小数精度要求
                val_accuracy = evaluate_with_precision(model, X_val, y_val, val_values, decimal_precision)
                
                # 记录最佳准确率
                best_accuracy = max(best_accuracy, val_accuracy)
            
            print(f'阶段 {stage_idx+1} 轮次 {epoch+1}/{cur_epochs}, '
                  f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, '
                  f'验证准确率: {val_accuracy:.4f}')
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 如果验证准确率达到高水平，提前结束阶段训练
            if val_loss < stage_losss_break or val_accuracy > 0.93:
                print(f"阶段 {stage_idx+1} 提前完成! 验证损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}")
                break
                
        # 保存阶段检查点
        torch.save(model.state_dict(), f"model_stage_{stage_idx+1}.pt")
        
        # 在阶段完成后加入混合训练(除第一阶段)
        if stage_idx >= 1 and stage_idx < len(stages) - 1:
            print(f"阶段{stage_idx+1}后进行混合训练...")
            
            # 准备混合数据
            mixed_X = []
            mixed_y = []
            mixed_values = []
            
            # 从已训练的阶段中各取一些样本
            for prev_X, prev_y, prev_values in stage_data:
                sample_size = min(500, len(prev_X))  # 每个阶段取500个样本
                mixed_X.append(prev_X[:sample_size])
                mixed_y.append(prev_y[:sample_size])
                mixed_values.extend(prev_values[:sample_size])
            
            mixed_X = torch.cat(mixed_X).to(device)
            mixed_y = torch.cat(mixed_y).to(device)
            
            # 混合训练15轮
            for mix_epoch in range(15):
                model.train()
                indices = torch.randperm(len(mixed_X))
                batch_size = 128
                total_loss = 0
                
                for i in range(0, len(mixed_X), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    inputs = mixed_X[batch_indices]
                    targets = mixed_y[batch_indices]
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                mix_loss = total_loss / (len(mixed_X) / batch_size)
                
                # 每5轮评估一次
                if mix_epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        mix_outputs = model(mixed_X)
                        mix_val_loss = loss_fn(mix_outputs, mixed_y).item()
                        mix_accuracy = evaluate_with_precision(model, mixed_X, mixed_y, mixed_values, decimal_precision=3)
                        print(f"混合训练 轮次 {mix_epoch+1}/15, 损失: {mix_loss:.4f}, 验证损失: {mix_val_loss:.4f}, 准确率: {mix_accuracy:.4f}")
        
        # 在新阶段开始时重置学习率
        if stage_transition_callback:
            stage_transition_callback(stage_idx)
        
    # 最终阶段：混合数据训练
    print("\n开始最终整合阶段: 混合数据训练")
    # 修改generate_mixed_data函数以返回原始值
    X_train, y_train, train_values = generate_mixed_data(num_samples=80000)  
    X_val, y_val, val_values = generate_mixed_data(num_samples=2000)
    
    # 将数据移到设备
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    batch_size = 1024
    
    # 最终训练
    for epoch in range(final_stage_epochs):
        # 训练模式
        model.train()
        total_loss = 0
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            inputs = X_train[batch_indices]
            targets = y_train[batch_indices]
            
            # 前向传播
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / (len(X_train) / batch_size)
        
        # 验证 - 使用最严格的精度要求
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_fn(val_outputs, y_val).item()
            
            # 使用最高精度要求评估
            val_accuracy = evaluate_with_precision(model, X_val, y_val, val_values, decimal_precision=4)
        
        print(f'最终阶段 轮次 {epoch+1}/{final_stage_epochs}, '
              f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, '
              f'验证准确率: {val_accuracy:.4f}')
        
        # 学习率调度
        scheduler.step(val_loss)
        
        if val_accuracy > final_accuracy_break:
            print(f"最终阶段 提前完成! 验证准确率: {val_accuracy:.4f}")
            break
    
    # 保存最终模型
    torch.save(model.state_dict(), "model_final.pt")
    return model

def evaluate_with_precision(model, X, y, original_values, decimal_precision):
    """根据指定的小数精度评估模型准确率"""
    correct_count = 0
    total_count = len(X)
    
    for i in range(total_count):
        a, b, c = original_values[i]
        
        # 按照指定小数位数进行加法验证
        factor = 10**decimal_precision
        a_truncated = round(a * factor) / factor
        b_truncated = round(b * factor) / factor
        c_truncated = round(c * factor) / factor
        
        # 验证加法是否正确，考虑指定小数位的精度
        correct_addition = abs((a_truncated + b_truncated) - c_truncated) < (0.5 * 10**(-decimal_precision))
        
        # 使用截断后的值生成特征并进行预测
        features = normalize_data_user(a_truncated, b_truncated, c_truncated, max_value=10000)
        input_tensor = torch.tensor([features], dtype=torch.float32).to(device)
        
        # 获取预测结果
        with torch.no_grad():
            output = model(input_tensor)
            predicted_prob = output[0, 1].item()  # 正确类别的概率
            model_prediction = predicted_prob > 0.5  # 1表示模型认为加法正确
        
        # 如果模型预测与加法正确性一致，计为正确
        if model_prediction == correct_addition:
            correct_count += 1
    
    return correct_count / total_count

def init_xavier(m):
    if isinstance(m, nn.Linear):
        # 使用Xavier均匀分布初始化权重
        nn.init.xavier_uniform_(m.weight)
        # 将偏置初始化为小的正数，避免ReLU神经元"死亡"
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

def run():
    # 创建模型并移到GPU
    model = SimpleAdditionNet().to(device)
    model.apply(init_xavier)
    criterion = nn.BCELoss()
    initial_lr = 0.001  # 保存初始学习率，用于重置
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    # 调整学习率调度器参数，更快响应变化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-6
    )
    
    # 修改阶段转换方式：在新阶段开始时重置学习率
    def stage_transition_callback(stage_idx):
        # 重置学习率为初始值的一半，帮助新阶段快速适应
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr * 0.5
    
    model = curriculum_training(
        model, criterion, optimizer, scheduler, 
        epochs_per_stage=100, 
        stage_losss_break=0.3, 
        final_accuracy_break=0.95,
        stage_transition_callback=stage_transition_callback
    )
    
    # 测试模型
    print("\n开始测试模型性能...")
    decimal_examples = [
        # 1-10：简单小数加法（正确结果）
        (12.34, 56.78, 69.12),       # 正确
        (123.45, 678.90, 802.35),    # 正确
        (0.75, 0.25, 1.00),          # 正确
        (9.99, 1.01, 11.00),         # 正确
        (45.67, 32.33, 78.00),       # 正确
        
        # 11-20：简单小数加法（错误结果）
        (22.22, 33.33, 56.66),       # 错误（正确应为55.55）
        (88.88, 11.11, 101.00),      # 错误（正确应为99.99）
        (7.50, 2.50, 9.50),          # 错误（正确应为10.00）
        (45.50, 45.50, 90.00),       # 错误（正确应为91.00）
        (67.89, 12.10, 81.99),       # 错误（正确应为79.99）
        
        # 21-30：大数加法（正确结果）
        (1234.56, 7890.12, 9124.68), # 正确
        (8765.43, 1234.57, 10000.00),# 正确
        (5432.10, 5432.10, 10864.20),# 正确
        (9999.99, 0.01, 10000.00),   # 正确
        (4567.89, 3210.11, 7778.00), # 正确
        
        # 31-40：大数加法（错误结果）
        (2345.67, 8901.23, 11000.00),# 错误（正确应为11246.90）
        (7777.77, 2222.22, 9999.00), # 错误（正确应为9999.99）
        (4444.44, 5555.55, 9999.00), # 错误（正确应为9999.99）
        (6789.01, 2345.67, 9035.68), # 错误（正确应为9134.68）
        (5000.00, 5000.00, 9000.00), # 错误（正确应为10000.00）
        
        # 41-50：更复杂的小数加法（正确结果）
        (123.456, 876.544, 1000.000),# 正确
        (0.1234, 0.8766, 1.0000),    # 正确
        (7.5432, 2.4568, 10.0000),   # 正确
        (99.9999, 0.0001, 100.0000), # 正确
        (567.123, 432.877, 1000.000),# 正确
        
        # 51-60：更复杂的小数加法（错误结果）
        (456.789, 543.211, 999.999), # 错误（正确应为1000.000）
        (1.2345, 2.3456, 3.5000),    # 错误（正确应为3.5801）
        (6.7890, 3.2109, 9.9998),    # 错误（正确应为9.9999）
        (0.0123, 0.0456, 0.0580),    # 错误（正确应为0.0579）
        (7.1234, 8.9876, 16.0000),   # 错误（正确应为16.1110）
        
        # 61-70：接近相等的数加法（正确结果）
        (999.999, 999.999, 1999.998),# 正确
        (555.555, 555.555, 1111.110),# 正确
        (123.123, 123.123, 246.246), # 正确
        (7.7777, 7.7777, 15.5554),   # 正确
        (0.5000, 0.5000, 1.0000),    # 正确
        
        # 71-80：接近相等的数加法（错误结果）
        (444.444, 444.444, 888.000), # 错误（正确应为888.888）
        (777.777, 777.777, 1555.555),# 错误（正确应为1555.554）
        (3.3333, 3.3333, 6.6000),    # 错误（正确应为6.6666）
        (1.1111, 1.1111, 2.2000),    # 错误（正确应为2.2222）
        (9.9999, 9.9999, 19.0000),   # 错误（正确应为19.9998）
        
        # 81-90：一个数很大一个数很小加法（正确结果）
        (9999.99, 0.01, 10000.00),   # 正确
        (0.0001, 999.9999, 1000.0000),# 正确
        (0.1234, 987.8766, 988.0000),# 正确
        (1.0000, 9999.0000, 10000.0000),# 正确
        (0.0001, 0.9999, 1.0000),    # 正确
        
        # 91-100：一个数很大一个数很小加法（错误结果）
        (9876.54, 0.01, 9876.54),    # 错误（正确应为9876.55）
        (0.0001, 100.0000, 100.0002),# 错误（正确应为100.0001）
        (1234.56, 0.001, 1234.56),   # 错误（正确应为1234.561）
        (0.0001, 9999.9998, 9999.0000),# 错误（正确应为9999.9999）
        (0.1111, 9999.8888, 9999.9000)# 错误（正确应为9999.9999）
    ]
    
    print(f"{'A':^10} + {'B':^10} = {'C':^10} | {'预测概率':^8} | {'实际情况':^8} | {'是否正确':^8}")
    print("-" * 60)
    
    # 统计检验结果与实际结果相符的次数与百分比
    total_examples = len(decimal_examples)
    correct_predictions = 0
    
    for a, b, c in decimal_examples:
        # 使用check_addition函数，它已经内部实现了加法验证逻辑
        prob = check_addition(model, a, b, c)
        
        # 确定小数保留位数
        if max(a, b, c) <= 10:
            decimal_places = 1
        elif max(a, b, c) <= 100:
            decimal_places = 2
        elif max(a, b, c) <= 1000:
            decimal_places = 3
        else:
            decimal_places = 4
        
        # 按精度要求验证加法
        factor = 10**decimal_places
        a_truncated = round(a * factor) / factor
        b_truncated = round(b * factor) / factor
        c_truncated = round(c * factor) / factor
        
        # 验证加法是否正确
        correct = abs((a_truncated + b_truncated) - c_truncated) < (0.5 * 10**(-decimal_places))
        prediction_correct = (correct and prob > 0.5) or (not correct and prob < 0.5)
        
        if prediction_correct:
            correct_predictions += 1
            
        print(f"{a:<10.4f} + {b:<10.4f} = {c:<10.4f} | {prob:>8.4f} | {'正确' if correct else '错误':^8} | {'✓' if prediction_correct else '✗':^8}")
    
    accuracy = correct_predictions / total_examples * 100
    print("\n检验结果统计:")
    print(f"总样本数: {total_examples}")
    print(f"检验结果与实际结果相符的次数: {correct_predictions}")
    print(f"检验准确率: {accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    model = run()
        