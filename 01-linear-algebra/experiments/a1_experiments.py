"""
A1 线性代数实验
验证点乘/叉乘/矩阵/旋转/齐次坐标的核心概念
需要: pip install numpy
"""
import numpy as np

# ============================================================
# 实验1：点乘 —— 投影长度 × 另一向量长度 = 方向相似度
# ============================================================
print("=" * 50)
print("实验1：点乘（Dot Product）")
print("=" * 50)

a = np.array([3.0, 1.0])
b = np.array([1.0, 2.0])

dot = np.dot(a, b)                          # 坐标公式
len_a, len_b = np.linalg.norm(a), np.linalg.norm(b)
cos_theta = dot / (len_a * len_b)           # 从点乘反推 cosθ
theta = np.arccos(cos_theta)                # 弧度
proj_len = dot / len_b                      # a 投影到 b 上的长度

print(f"a={a}, b={b}")
print(f"a·b (坐标公式) = {dot:.4f}")
print(f"|a|={len_a:.4f}, |b|={len_b:.4f}")
print(f"a·b = |a||b|cosθ → cosθ = {cos_theta:.4f} → θ = {np.degrees(theta):.1f}°")
print(f"投影长度 (a在b上) = {proj_len:.4f}")
print(f"验证：投影长度×|b| = {proj_len * len_b:.4f} ≈ a·b = {dot:.4f}")
print(f"结论：点乘>0 → 夹角<90°同向；点乘=0 → 垂直；点乘<0 → 反向")
print()

# 游戏应用：视野判断
forward = np.array([0.0, 1.0])              # 面朝方向（正前方）
enemy_dir = np.array([0.5, 0.8])            # 敌人相对方位
fov_dot = np.dot(forward, enemy_dir)
print(f"【游戏实例】面朝={forward}，敌人方位={enemy_dir}")
print(f"  点乘={fov_dot:.4f} → {'前方，可见' if fov_dot > 0 else '身后，不可见'}")

# ============================================================
# 实验2：叉乘 —— 垂直于两向量的新向量，长度=平行四边形面积
# ============================================================
print("\n" + "=" * 50)
print("实验2：叉乘（Cross Product）")
print("=" * 50)

a3 = np.array([3.0, 0.0, 0.0])
b3 = np.array([0.0, 4.0, 0.0])

cross = np.cross(a3, b3)
# 叉乘长度 = |a||b|sinθ = 平行四边形面积
area = np.linalg.norm(cross)
expected_area = 3 * 4                       # 矩形面积

print(f"a={a3}, b={b3}")
print(f"a×b = {cross}  ← 垂直于ab平面（Z轴）")
print(f"|a×b| = {area:.4f} = 平行四边形面积")
print(f"验证：矩形 3×4 = {expected_area}，匹配！")
print(f"验证右手定则：食指a沿X，中指b沿Y → 拇指={cross}（Z轴正方向）[OK]")
print(f"反交换律：b×a = {np.cross(b3, a3)} = -(a×b)")
print()

# 游戏应用：法线计算
tri = np.array([[1,0,0], [0,1,0], [0,0,1]])  # 三角形三个顶点
edge1 = tri[1] - tri[0]
edge2 = tri[2] - tri[0]
normal = np.cross(edge1, edge2)
normal = normal / np.linalg.norm(normal)       # 归一化
print(f"【游戏实例】三角形两条边叉乘 = 法线方向 {normal}")

# ============================================================
# 实验3：矩阵×向量 = 用变换后的基向量重新组合
# ============================================================
print("\n" + "=" * 50)
print("实验3：矩阵×向量 —— 基向量变换视角")
print("=" * 50)

M = np.array([[2, 1],                        # 第一列=变换后的i
              [0, 3]])                       # 第二列=变换后的j
v = np.array([1, 1])

result = M @ v                               # 矩阵×向量
manual = v[0] * M[:, 0] + v[1] * M[:, 1]    # x份新i + y份新j

print(f"M = \n{M}")
print(f"M 的第1列（变换后的i）= {M[:, 0]}")
print(f"M 的第2列（变换后的j）= {M[:, 1]}")
print(f"v = {v}")
print(f"M × v = {result}")
print(f"手动：{v[0]}×(变换后i) + {v[1]}×(变换后j) = {manual}")
print("结论：矩阵乘向量 = 用变换后的基向量重新组合该点 [OK]")

# ============================================================
# 实验4：2D旋转矩阵验证
# ============================================================
print("\n" + "=" * 50)
print("实验4：2D旋转矩阵")
print("=" * 50)

theta = np.radians(90)                       # 旋转90度
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])                 # 初始在X轴上的点
rotated = R @ point

print(f"旋转角度：{np.degrees(theta):.0f}°")
print(f"旋转矩阵 R =\n{R}")
print(f"初始点 {point} → 旋转后 {np.round(rotated, 4)}")
print(f"验证：(1,0) 逆时针90° → (0,1) [OK]")
print(f"验证 det(R) = {np.linalg.det(R):.4f}（=1，面积不变）")

# 旋转矩阵两列是正交的
print(f"列0·列1 = {np.dot(R[:,0], R[:,1]):.4f}（=0，互相垂直）")
print(f"|列0| = {np.linalg.norm(R[:,0]):.4f}, |列1| = {np.linalg.norm(R[:,1]):.4f}（都是1）")

# ============================================================
# 实验5：3D旋转矩阵 —— 绕哪个轴，那个基向量不动
# ============================================================
print("\n" + "=" * 50)
print("实验5：3D旋转矩阵（绕X/Y/Z轴）")
print("=" * 50)

angle = np.radians(90)
c, s = np.cos(angle), np.sin(angle)

Rx = np.array([[1, 0,  0],
               [0, c, -s],
               [0, s,  c]])
Ry = np.array([[ c, 0, s],
               [ 0, 1, 0],
               [-s, 0, c]])
Rz = np.array([[c, -s, 0],
               [s,  c, 0],
               [0,  0, 1]])

# 绕X轴旋转：X列(1,0,0)不变
print(f"Rx 第1列（原X轴）= {Rx[:, 0]}  ← 不变")
print(f"Ry 第2列（原Y轴）= {Ry[:, 1]}  ← 不变")
print(f"Rz 第3列（原Z轴）= {Rz[:, 2]}  ← 不变")
print("结论：旋转轴对应的基向量不动，另外两个在各自平面里转 [OK]")

# ============================================================
# 实验6：旋转 vs 平移，顺序不同结果不同
# ============================================================
print("\n" + "=" * 50)
print("实验6：旋转与平移的顺序")
print("=" * 50)

point = np.array([1.0, 0.0])
R90 = np.array([[0, -1],
                [1,  0]])                    # 90°旋转矩阵
T = np.array([2.0, 0.0])                    # 向右平移2

rotate_then_translate = R90 @ point + T      # 先旋转再平移
translate_then_rotate = R90 @ (point + T)    # 先平移再旋转

print(f"原始点: {point}")
print(f"先旋转再平移: {rotate_then_translate}  （点(1,0)→旋转到(0,1)→平移→(2,1)）")
print(f"先平移再旋转: {translate_then_rotate}（点(1,0)→平移到(3,0)→旋转→(0,3)）")
print("结论：结果完全不同！因为旋转始终绕原点进行 [OK]")

# ============================================================
# 实验7：4×4 齐次坐标 —— 让平移变成矩阵乘法
# ============================================================
print("\n" + "=" * 50)
print("实验7：4×4齐次坐标")
print("=" * 50)

# 3×3无法做平移
M33 = np.array([[2, 0, 0],
                [0, 2, 0],
                [0, 0, 2]])
v3 = np.array([1.0, 2.0, 3.0])
print(f"3×3 缩放: {M33 @ v3}  （只能缩放/旋转，不能平移）")

# 4×4齐次坐标 — 平移塞进最后一列
M44 = np.array([[1, 0, 0, 5],                # 平移(5,3,2)的矩阵
                [0, 1, 0, 3],
                [0, 0, 1, 2],
                [0, 0, 0, 1]])
v4 = np.array([1.0, 2.0, 3.0, 1.0])         # w=1 表示这是一个点
result4 = M44 @ v4
print(f"4×4 平移: ({v4[0]},{v4[1]},{v4[2]}) → ({result4[0]},{result4[1]},{result4[2]})")
print(f"验证: (1+5, 2+3, 3+2) = (6,5,5) [OK]")

# w=0 表示方向（不受平移影响，方向不需要平移）
dir4 = np.array([0.0, 0.0, 1.0, 0.0])       # w=0
result_dir = M44 @ dir4
print(f"方向向量 w=0: {dir4[:3]} → {result_dir[:3]}（方向不受平移影响 [OK]）")

# 组合变换：先缩放再旋转再平移
S = np.array([[2, 0, 0, 0],
              [0, 2, 0, 0],
              [0, 0, 2, 0],
              [0, 0, 0, 1]])
Rz44 = np.array([[c, -s, 0, 0],
                 [s,  c, 0, 0],
                 [0,  0, 1, 0],
                 [0,  0, 0, 1]])
T44 = np.array([[1, 0, 0, 5],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

composite = T44 @ Rz44 @ S                  # 从右往左：先缩放→旋转→平移
p = np.array([1.0, 1.0, 0.0, 1.0])
transformed = composite @ p
print(f"\n组合变换 T·R·S @ (1,1,0) = {transformed}")
print("这就是引擎里 Model Matrix 的原理 [OK]")

print("\n" + "=" * 50)
print("A1 全部实验完成！")
print("=" * 50)
