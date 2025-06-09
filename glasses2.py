import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, VBox, HBox, interactive_output, Layout, ToggleButtons, Button
import ipywidgets as widgets
import random # randomモジュールを追加

# パーツを描画する汎用関数（回転・拡大・位置対応、幅調整対応）
def draw_shape(shape, size=1.0, width_ratio=1.0, angle=0.0, center=(0, 0),
               fill_color=None, alpha=1.0, edgecolor=None, linewidth=0, fill=True, zorder=1): # zorder引数を追加
    angle_rad = np.deg2rad(angle)
    x_coords, y_coords = [], []

    if shape == 'Circle':
        ellipse = Ellipse(xy=center, width=size * 2 * width_ratio, height=size * 2, angle=angle,
                          fill=fill, facecolor=fill_color if fill else 'none', alpha=alpha if fill else 1.0,
                          edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
        return ellipse
    elif shape == 'Square':
        width = size * 2 * width_ratio
        height = size * 2

        corners_local = np.array([
            [-width/2, height/2],  # Top-left
            [width/2, height/2],   # Top-right
            [width/2, -height/2],  # Bottom-right
            [-width/2, -height/2] # Bottom-left
        ])

        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        rotated_points = (rot_matrix @ corners_local.T).T

        x_coords = rotated_points[:, 0] + center[0]
        y_coords = rotated_points[:, 1] + center[1]

        if fill or edgecolor is not None:
            polygon = plt.Polygon(np.array([x_coords, y_coords]).T,
                                  closed=True,
                                  facecolor=fill_color if fill else 'none', alpha=alpha if fill else 1.0,
                                  edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
            return polygon

        return x_coords, y_coords

    elif shape == 'Triangle':
        points = np.array([
            [0, size * (2/3)],
            [size * width_ratio * 0.5, -size * (1/3)],
            [-size * width_ratio * 0.5, -size * (1/3)],
        ])

        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_points = (rot_matrix @ points.T).T

        x_coords = rotated_points[:, 0] + center[0]
        y_coords = rotated_points[:, 1] + center[1]

        if fill or edgecolor is not None:
            polygon = plt.Polygon(np.array([x_coords, y_coords]).T,
                                  closed=True,
                                  facecolor=fill_color if fill else 'none', alpha=alpha if fill else 1.0,
                                  edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
            return polygon

        return x_coords, y_coords

    return [], []

# 眉毛を描画する関数（線分）
def draw_brow_line(length, angle=0.0, thickness=2, center=(0, 0), color='black', zorder=1): # zorder引数を追加
    angle_rad = np.deg2rad(angle)

    half_length = length / 2.0

    x_start_local = -half_length
    y_start_local = 0
    x_end_local = half_length
    y_end_local = 0

    x_start_rot = np.cos(angle_rad) * x_start_local - np.sin(angle_rad) * y_start_local
    y_start_rot = np.sin(angle_rad) * x_start_local + np.cos(angle_rad) * y_start_local
    x_end_rot = np.cos(angle_rad) * x_end_local - np.sin(angle_rad) * y_end_local
    y_end_rot = np.sin(angle_rad) * x_end_local + np.cos(angle_rad) * y_end_local

    x_coords = [x_start_rot + center[0], x_end_rot + center[0]]
    y_coords = [y_start_rot + center[1], y_end_rot + center[1]]

    return x_coords, y_coords, zorder


# 耳を描画する関数（顔の輪郭から伸びる曲線）
def draw_ear_from_face(start_point, end_point, control_point_distance=1.0, side='left', ear_rotation=0.0, color='black', linewidth=2, zorder=1): # zorder引数を追加
    num_points = 100
    t = np.linspace(0, 1, num_points)

    mid_x = (start_point[0] + end_point[0]) / 2.0
    mid_y = (start_point[1] + end_point[1]) / 2.0

    vec_x_connect = start_point[0] - end_point[0]
    vec_y_connect = start_point[1] - end_point[1]

    if side == 'left':
        normal_vec_x = -vec_y_connect
        normal_vec_y = vec_x_connect
    else: # right
        normal_vec_x = vec_y_connect
        normal_vec_y = -vec_x_connect

    norm = np.sqrt(normal_vec_x**2 + normal_vec_y**2)
    if norm > 0:
        normal_vec_x /= norm
        normal_vec_y /= norm
    else:
        normal_vec_x = 1 if side == 'right' else -1
        normal_vec_y = 0

    control_x_base = mid_x + normal_vec_x * control_point_distance
    control_y_base = mid_y + normal_vec_y * control_point_distance

    rotation_rad = np.deg2rad(ear_rotation)

    rel_cx = control_x_base - mid_x
    rel_cy = control_y_base - mid_y

    rotated_rel_cx = np.cos(rotation_rad) * rel_cx - np.sin(rotation_rad) * rel_cy
    rotated_rel_cy = np.sin(rotation_rad) * rel_cx + np.cos(rotation_rad) * rel_cy

    final_control_x = mid_x + rotated_rel_cx
    final_control_y = mid_y + rotated_rel_cy

    x_curve = (1-t)**2 * start_point[0] + 2*(1-t)*t * final_control_x + t**2 * end_point[0]
    y_curve = (1-t)**2 * start_point[1] + 2*(1-t)*t * final_control_y + t**2 * end_point[1]

    return x_curve, y_curve, zorder

# レンズの縁の点を取得する関数を再々々修正
def get_edge_point(shape, size, width_ratio, lens_angle_deg, center, side_of_face):
    lens_angle_rad = np.deg2rad(lens_angle_deg)

    if shape == 'Circle': # Ellipseとして処理
        a = size * width_ratio  # 半径X
        b = size                # 半径Y

        angles_for_eval = np.linspace(0, 2*np.pi, 360, endpoint=False) # 360点評価

        local_points = np.array([
            [a * np.cos(t), b * np.sin(t)] for t in angles_for_eval
        ])

        rot_matrix = np.array([
            [np.cos(lens_angle_rad), -np.sin(lens_angle_rad)],
            [np.sin(lens_angle_rad), np.cos(lens_angle_rad)]
        ])
        rotated_points = (rot_matrix @ local_points.T).T

        if side_of_face == 'left_lens': # 左レンズは右側（Xが大きい方）が鼻側
            idx = np.argmax(rotated_points[:, 0])
        else: # right_lens 右レンズは左側（Xが小さい方）が鼻側
            idx = np.argmin(rotated_points[:, 0])

        edge_point = np.array([rotated_points[idx, 0] + center[0], rotated_points[idx, 1] + center[1]])

    elif shape == 'Square': # Rectangleとして処理
        width = size * 2 * width_ratio
        height = size * 2

        corners_local = np.array([
            [-width/2, height/2],  # Top-left
            [width/2, height/2],   # Top-right
            [width/2, -height/2],  # Bottom-right
            [-width/2, -height/2]  # Bottom-left
        ])

        rot_matrix = np.array([
            [np.cos(lens_angle_rad), -np.sin(lens_angle_rad)],
            [np.sin(lens_angle_rad), np.cos(lens_angle_rad)]
        ])
        rotated_corners = (rot_matrix @ corners_local.T).T

        if side_of_face == 'left_lens':
            idx = np.argmax(rotated_corners[:, 0])
        else:
            idx = np.argmin(rotated_corners[:, 0])

        edge_point = np.array([rotated_corners[idx, 0] + center[0], rotated_corners[idx, 1] + center[1]])

    elif shape == 'Triangle':
        points = np.array([
            [0, size * (2/3)],
            [size * width_ratio * 0.5, -size * (1/3)],
            [-size * width_ratio * 0.5, -size * (1/3)],
        ])

        rot_matrix = np.array([
            [np.cos(lens_angle_rad), -np.sin(lens_angle_rad)],
            [np.sin(lens_angle_rad), np.cos(lens_angle_rad)]
        ])
        rotated_points = (rot_matrix @ points.T).T

        if side_of_face == 'left_lens':
            idx = np.argmax(rotated_points[:, 0])
        else:
            idx = np.argmin(rotated_points[:, 0])

        edge_point = np.array([rotated_points[idx, 0] + center[0], rotated_points[idx, 1] + center[1]])

    else:
        edge_point = center
    return edge_point

# --- サプライズ計算のための関数とデータ構造 ---

# パラメータの離散化と事前分布の定義
# 各パラメータに対して、min, max, step から10個の区間を定義し、事前確率を割り当てる

# 例：レンズの大きさ (left_size, right_size) の事前分布
# 平均的なサイズ (1.0) の確率を高く、離れるほど低くする
def define_discrete_prior(min_val, max_val, step, num_bins=10, center_value=None):
    if center_value is None:
        center_value = (min_val + max_val) / 2.0

    bins = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # ガウス分布に近い形で確率を割り当てる例
    # 標準偏差を調整して、分布の「広がり」を制御
    std_dev = (max_val - min_val) / 6 # 6シグマで全範囲をカバーする目安
    probabilities = np.exp(-0.5 * ((bin_centers - center_value) / std_dev)**2)
    probabilities /= np.sum(probabilities) # 確率の合計を1に正規化

    return bin_centers, probabilities

# 形状の事前分布 (カテゴリカル)
shape_prior = {
    'Circle': 0.6,
    'Square': 0.2,
    'Triangle': 0.2,
}
# 確率の合計を1に正規化 (念のため)
total_shape_prior = sum(shape_prior.values())
for shape, prob in shape_prior.items():
    shape_prior[shape] = prob / total_shape_prior


# 各パラメータの事前分布を保持する辞書
# key: パラメータ名, value: (bin_centers, probabilities) または (categorical_probabilities_dict)
prior_distributions = {
    'left_size': define_discrete_prior(0.5, 2.0, 0.1, num_bins=10, center_value=1.0),
    'right_size': define_discrete_prior(0.5, 2.0, 0.1, num_bins=10, center_value=1.0),
    'left_width_ratio': define_discrete_prior(0.5, 2.0, 0.05, num_bins=10, center_value=1.0),
    'right_width_ratio': define_discrete_prior(0.5, 2.0, 0.05, num_bins=10, center_value=1.0),
    'left_angle': define_discrete_prior(-180, 180, 1, num_bins=10, center_value=0.0),
    'right_angle': define_discrete_prior(-180, 180, 1, num_bins=10, center_value=0.0),
    'lens_red': define_discrete_prior(0.0, 1.0, 0.01, num_bins=10, center_value=0.0), # 透明なレンズに近い色を優先
    'lens_green': define_discrete_prior(0.0, 1.0, 0.01, num_bins=10, center_value=0.0),
    'lens_blue': define_discrete_prior(0.0, 1.0, 0.01, num_bins=10, center_value=0.0),
    'lens_alpha': define_discrete_prior(0.0, 1.0, 0.01, num_bins=10, center_value=0.5), # 半透明を標準とする
    'red': define_discrete_prior(0.0, 1.0, 0.01, num_bins=10, center_value=0.0), # 黒フレームを優先
    'green': define_discrete_prior(0.0, 1.0, 0.01, num_bins=10, center_value=0.0),
    'blue': define_discrete_prior(0.0, 1.0, 0.01, num_bins=10, center_value=0.0),
    'thickness': define_discrete_prior(1, 10, 1, num_bins=10, center_value=2), # 細めを標準とする
    'left_shape': shape_prior, # カテゴリカルな事前分布をそのまま保持
    'right_shape': shape_prior, # カテゴリカルな事前分布をそのまま保持
}

# パラメータ値から対応する離散ビンと確率を取得するヘルパー関数
def get_probability_from_prior(param_name, value):
    prior = prior_distributions.get(param_name)
    if prior is None:
        return 1.0 # 事前分布がない場合は確率1とする（サプライズなし）

    if isinstance(prior, dict): # カテゴリカルな事前分布の場合
        return prior.get(value, 1e-9) # 辞書から直接確率を取得、ない場合はごく小さい値 (Log(0)回避)
    else: # 連続値の事前分布の場合
        bin_centers, probabilities = prior
        # 最も近いビンを探す
        idx = np.argmin(np.abs(bin_centers - value))
        return probabilities[idx]

# サプライズを計算する関数
def calculate_surprise(observed_params):
    total_log_likelihood = 0.0
    for param_name, observed_value in observed_params.items():
        prob = get_probability_from_prior(param_name, observed_value)
        # 確率が0の場合のlog(0)エラーを避けるため、極小値でクリップ
        prob = max(prob, 1e-9) # 最小確率を非常に小さい正の値に設定
        total_log_likelihood += np.log(prob) # 対数尤度を加算

    surprise = -total_log_likelihood # 負の対数尤度がサプライズ
    return surprise


# --- 新しい機能: サプライズ目標値に基づくランダムデザイン生成 ---

# サンプリング対象となるメガネ関連のパラメータ名リスト
GLASSES_PARAM_NAMES = [
    'left_shape', 'right_shape',
    'left_size', 'right_size',
    'left_width_ratio', 'right_width_ratio',
    'left_angle', 'right_angle',
    'lens_red', 'lens_green', 'lens_blue', 'lens_alpha',
    'red', 'green', 'blue', 'thickness'
]

def generate_random_param_value(param_name, target_surprise_level):
    """
    指定されたパラメータについて、目標サプライズレベルに基づいてランダムな値を生成する。
    target_surprise_level: 0 (低いサプライズ) から 1 (高いサプライズ) の範囲。
    """
    prior = prior_distributions.get(param_name)
    if prior is None:
        # 事前分布がない場合は、スライダーの全範囲から一様にサンプリング
        widget_ref = all_controls[param_name]
        if isinstance(widget_ref, FloatSlider):
            return random.uniform(widget_ref.min, widget_ref.max)
        elif isinstance(widget_ref, IntSlider):
            return random.randint(widget_ref.min, widget_ref.max)
        elif isinstance(widget_ref, Dropdown):
            return random.choice(widget_ref.options)
        return None # 未定義のウィジェットタイプ

    if isinstance(prior, dict): # カテゴリカルな事前分布の場合
        items = list(prior.items())
        values = [item[0] for item in items]
        probabilities = np.array([item[1] for item in items])

        if target_surprise_level < 0.3: # 低いサプライズ: 事前分布に従う
            return np.random.choice(values, p=probabilities)
        elif target_surprise_level < 0.7: # 中程度のサプライズ: 少し偏りを変える
            # 事前分布の逆を重みとするが、完全に逆転させると極端になるので、線形補間
            # 例えば、prob_inverse = 1 - prob / max_prob (正規化が必要)
            # よりシンプルに、少し一様に近づける
            mix_factor = (target_surprise_level - 0.3) / 0.4 # 0から1の範囲
            mixed_probs = (1 - mix_factor) * probabilities + mix_factor * (1.0 / len(values))
            mixed_probs /= np.sum(mixed_probs)
            return np.random.choice(values, p=mixed_probs)
        else: # 高いサプライズ: 一様に近いサンプリング (予測外の形状が出やすい)
            # 確率の低いものが出やすいように逆数を重みとする
            inverse_probs = 1.0 / (probabilities + 1e-9) # 0割り防止
            inverse_probs /= np.sum(inverse_probs)
            return np.random.choice(values, p=inverse_probs)

    else: # 連続値の事前分布の場合
        bin_centers, probabilities = prior

        if target_surprise_level < 0.3: # 低いサプライズ: 事前分布に従う
            chosen_bin_idx = np.random.choice(len(bin_centers), p=probabilities)
        elif target_surprise_level < 0.7: # 中程度のサプライズ: 少し偏りを変える
            mix_factor = (target_surprise_level - 0.3) / 0.4 # 0から1の範囲
            mixed_probs = (1 - mix_factor) * probabilities + mix_factor * (1.0 / len(bin_centers)) # 一様分布と線形補間
            mixed_probs /= np.sum(mixed_probs)
            chosen_bin_idx = np.random.choice(len(bin_centers), p=mixed_probs)
        else: # 高いサプライズ: 確率の低いものが選ばれやすいようにする (予測外の値が出やすい)
            # 確率の逆数を重みとして使用
            inverse_probs = 1.0 / (probabilities + 1e-9) # 0割り防止
            inverse_probs /= np.sum(inverse_probs)
            chosen_bin_idx = np.random.choice(len(bin_centers), p=inverse_probs)

        # 選択されたビンの中心値の周りで少しランダム性を加える
        # スライダーのステップ幅を考慮して微調整
        bin_min = (prior[0][chosen_bin_idx] - (prior[0][1] - prior[0][0]) / 2) if chosen_bin_idx > 0 else prior[0][0] # ビンの最小値
        bin_max = (prior[0][chosen_bin_idx] + (prior[0][1] - prior[0][0]) / 2) if chosen_bin_idx < len(prior[0]) -1 else prior[0][-1] # ビンの最大値
        
        # 実際にそのビンの範囲内で一様乱数を生成
        value = random.uniform(max(bin_min, prior[0][0] - (prior[0][1] - prior[0][0]) / 2), # min_valより小さくならないようにクリップ
                                min(bin_max, prior[0][-1] + (prior[0][1] - prior[0][0]) / 2)) # max_valより大きくならないようにクリップ
        
        # スライダーのステップに合わせて丸める
        widget_ref = all_controls[param_name]
        if isinstance(widget_ref, FloatSlider):
            value = round(value / widget_ref.step) * widget_ref.step
            value = np.clip(value, widget_ref.min, widget_ref.max) # スライダーの範囲にクリップ
        elif isinstance(widget_ref, IntSlider):
            value = round(value)
            value = int(np.clip(value, widget_ref.min, widget_ref.max)) # スライダーの範囲にクリップ

        return value

def generate_and_set_random_glasses_design(target_surprise_level):
    """
    目標サプライズレベルに基づいてランダムなメガネデザインを生成し、ウィジェットに適用する。
    """
    generated_params = {}
    for param_name in GLASSES_PARAM_NAMES:
        generated_params[param_name] = generate_random_param_value(param_name, target_surprise_level)
    
    # 生成されたパラメータをウィジェットに設定
    for param_name, value in generated_params.items():
        if param_name in all_controls:
            # Dropdownはvalueの型が違うので注意
            if isinstance(all_controls[param_name], widgets.Dropdown):
                if value in all_controls[param_name].options:
                    all_controls[param_name].value = value
            else:
                all_controls[param_name].value = value
        
# 描画関数を更新し、サプライズ計算と表示を追加
def draw_face_and_glasses(
    face_shape='Circle', face_size=3.0, face_width_ratio=1.0,

    selected_part='face',

    # 左右共通のオフセット
    brow_offset_x=0.0, brow_offset_y=0.0,

    eye_shape='Circle',
    eye_size=0.3, eye_width_ratio=1.0, eye_angle=0.0,
    eye_offset_x=0.0, eye_offset_y=0.0,

    brow_length=1.0, brow_angle=0.0, brow_thickness=2,

    nose_shape='Triangle', nose_size=0.5, nose_width_ratio=1.0,
    nose_offset_y=0.0,

    mouth_shape='Circle', mouth_size=0.7, mouth_width_ratio=1.0, mouth_angle=0.0,
    mouth_offset_y=0.0,

    ear_control_distance=1.0, ear_y_attach_offset=0.0, ear_span=1.5, ear_rotation=0.0,

    left_shape='Circle', right_shape='Circle',
    left_size=1.0, right_size=1.0,
    left_width_ratio=1.0, right_width_ratio=1.0,
    left_angle=0.0, right_angle=0.0,
    red=0.0, green=0.0, blue=0.0, # フレームの色
    thickness=2,
    lens_red=0.0, lens_green=0.0, lens_blue=0.0, lens_alpha=0.5
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    frame_color = (red, green, blue)
    lens_fill_color = (lens_red, lens_green, lens_blue)

    # 顔輪郭
    face_obj = draw_shape(face_shape, face_size, face_width_ratio, 0.0, (0, 0), edgecolor='black', linewidth=2, fill=False, zorder=1)
    if isinstance(face_obj, (Ellipse, Rectangle, plt.Polygon)):
        ax.add_patch(face_obj)
    else:
        ax.plot(face_obj[0], face_obj[1], color='black', linewidth=2, zorder=1)

    # 両目
    eye_L_base_center = np.array([-2, 1])
    eye_R_base_center = np.array([2, 1])

    eye_L_center = eye_L_base_center + np.array([eye_offset_x, eye_offset_y])
    eye_R_center = eye_R_base_center + np.array([-eye_offset_x, eye_offset_y])

    eye_L_angle = eye_angle
    eye_R_angle = -eye_angle

    eye_L_obj = draw_shape(eye_shape, eye_size, eye_width_ratio, eye_L_angle, eye_L_center, edgecolor='black', linewidth=2, fill=False, zorder=2)
    eye_R_obj = draw_shape(eye_shape, eye_size, eye_width_ratio, eye_R_angle, eye_R_center, edgecolor='black', linewidth=2, fill=False, zorder=2)

    if isinstance(eye_L_obj, (Ellipse, Rectangle, plt.Polygon)):
        ax.add_patch(eye_L_obj)
        ax.add_patch(eye_R_obj)
    else:
        ax.plot(eye_L_obj[0], eye_L_obj[1], color='black', linewidth=2, zorder=2)
        ax.plot(eye_R_obj[0], eye_R_obj[1], color='black', linewidth=2, zorder=2)

    # 両眉
    brow_L_base_center = np.array([-2, 2])
    brow_R_base_center = np.array([2, 2])

    brow_L_center = brow_L_base_center + np.array([brow_offset_x, brow_offset_y])
    brow_R_center = brow_R_base_center + np.array([-brow_offset_x, brow_offset_y])

    brow_R_angle = -brow_angle

    x, y, z = draw_brow_line(brow_length, brow_angle, brow_thickness, brow_L_center, color='black', zorder=2)
    ax.plot(x, y, color='black', linewidth=brow_thickness, zorder=z)
    x, y, z = draw_brow_line(brow_length, brow_R_angle, brow_thickness, brow_R_center, color='black', zorder=2)
    ax.plot(x, y, color='black', linewidth=brow_thickness, zorder=z)

    # 鼻
    nose_base_center = (0, 0.5)
    nose_center = (nose_base_center[0], nose_base_center[1] + nose_offset_y)
    nose_obj = draw_shape(nose_shape, nose_size, nose_width_ratio, 0.0, nose_center, edgecolor='black', linewidth=2, fill=False, zorder=2)
    if isinstance(nose_obj, (Ellipse, Rectangle, plt.Polygon)):
        ax.add_patch(nose_obj)
    else:
        ax.plot(nose_obj[0], nose_obj[1], color='black', linewidth=2, zorder=2)

    # メガネレンズの中心は目の中心に合わせる
    left_lens_center = eye_L_center
    right_lens_center = eye_R_center

    # レンズの内側部分を描画（透明度ありで塗りつぶし、縁なし）
    # zorderを低めに設定し、フレームより下に描画
    left_lens_fill_obj = draw_shape(left_shape, left_size, left_width_ratio, left_angle, left_lens_center,
                                    fill_color=lens_fill_color, alpha=lens_alpha,
                                    edgecolor='none', linewidth=0, fill=True, zorder=3)
    if isinstance(left_lens_fill_obj, (Ellipse, Rectangle, plt.Polygon)):
        ax.add_patch(left_lens_fill_obj)

    right_lens_fill_obj = draw_shape(right_shape, right_size, right_width_ratio, right_angle, right_lens_center,
                                     fill_color=lens_fill_color, alpha=lens_alpha,
                                     edgecolor='none', linewidth=0, fill=True, zorder=3)
    if isinstance(right_lens_fill_obj, (Ellipse, Rectangle, plt.Polygon)):
        ax.add_patch(right_lens_fill_obj)

    # フレーム部分を描画（塗りつぶしなし）
    # zorderをレンズより高く設定し、レンズの上に描画
    left_frame_obj = draw_shape(left_shape, left_size, left_width_ratio, left_angle, left_lens_center,
                                edgecolor=frame_color, linewidth=thickness, fill=False, zorder=4)
    if isinstance(left_frame_obj, (Ellipse, Rectangle, plt.Polygon)):
        ax.add_patch(left_frame_obj)
    else:
        ax.plot(left_frame_obj[0], left_frame_obj[1], color=frame_color, linewidth=thickness, zorder=4)

    right_frame_obj = draw_shape(right_shape, right_size, right_width_ratio, right_angle, right_lens_center,
                                 edgecolor=frame_color, linewidth=thickness, fill=False, zorder=4)
    if isinstance(right_frame_obj, (Ellipse, Rectangle, plt.Polygon)):
        ax.add_patch(right_frame_obj)
    else:
        ax.plot(right_frame_obj[0], right_frame_obj[1], color=frame_color, linewidth=thickness, zorder=4)

    # ブリッジの接続点を計算（これはフレームのサイズで計算）
    left_bridge_point = get_edge_point(left_shape, left_size, left_width_ratio, left_angle, left_lens_center, 'left_lens')
    right_bridge_point = get_edge_point(right_shape, right_size, right_width_ratio, right_angle, right_lens_center, 'right_lens')

    # ブリッジ描画
    bridge_y = left_lens_center[1]

    ax.plot([left_bridge_point[0], right_bridge_point[0]],
            [bridge_y, bridge_y],
            color=frame_color, linewidth=thickness, zorder=4) # フレームの色と太さを適用, zorderを高く設定

    # 口
    mouth_base_center = (0, -1.5)
    mouth_center = (mouth_base_center[0], mouth_base_center[1] + mouth_offset_y)
    mouth_obj = draw_shape(mouth_shape, mouth_size, mouth_width_ratio, mouth_angle, mouth_center, edgecolor='black', linewidth=2, fill=False, zorder=2)
    if isinstance(mouth_obj, (Ellipse, Rectangle, plt.Polygon)):
        ax.add_patch(mouth_obj)
    else:
        ax.plot(mouth_obj[0], mouth_obj[1], color='black', linewidth=2, zorder=2)

    # 両耳
    ear_top_y = ear_y_attach_offset + ear_span / 2.0
    ear_bottom_y = ear_y_attach_offset - ear_span / 2.0

    clipped_top_y = np.clip(ear_top_y, -face_size, face_size)
    clipped_bottom_y = np.clip(ear_bottom_y, -face_size, face_size)

    face_radius_x = face_size * face_width_ratio
    face_radius_y = face_size

    def get_face_x_at_y(y, r_x, r_y):
        if abs(y) > r_y:
            return 0
        return r_x * np.sqrt(max(0, 1 - (y/r_y)**2))

    attach_x_top_L = -get_face_x_at_y(clipped_top_y, face_radius_x, face_radius_y)
    attach_x_bottom_L = -get_face_x_at_y(clipped_bottom_y, face_radius_x, face_radius_y)

    attach_x_top_R = get_face_x_at_y(clipped_top_y, face_radius_x, face_radius_y)
    attach_x_bottom_R = get_face_x_at_y(clipped_bottom_y, face_radius_x, face_radius_y)

    start_point_L = (attach_x_top_L, clipped_top_y)
    end_point_L = (attach_x_bottom_L, clipped_bottom_y)

    start_point_R = (attach_x_top_R, clipped_top_y)
    end_point_R = (attach_x_bottom_R, clipped_bottom_y)


    final_ear_rotation_L = ear_rotation
    final_ear_rotation_R = -ear_rotation

    x, y, z = draw_ear_from_face(start_point_L, end_point_L, ear_control_distance, side='left', ear_rotation=final_ear_rotation_L, color='black', linewidth=2, zorder=1)
    ax.plot(x, y, color='black', linewidth=2, zorder=z)

    x, y, z = draw_ear_from_face(start_point_R, end_point_R, ear_control_distance, side='right', ear_rotation=final_ear_rotation_R, color='black', linewidth=2, zorder=z)
    ax.plot(x, y, color='black', linewidth=2, zorder=z)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-4, 5)

    # --- サプライズ計算と表示 ---
    # 現在のメガネ関連パラメータを収集
    observed_glasses_params = {
        'left_shape': left_shape,
        'right_shape': right_shape,
        'left_size': left_size,
        'right_size': right_size,
        'left_width_ratio': left_width_ratio,
        'right_width_ratio': right_width_ratio,
        'left_angle': left_angle,
        'right_angle': right_angle,
        'lens_red': lens_red,
        'lens_green': lens_green,
        'lens_blue': lens_blue,
        'lens_alpha': lens_alpha,
        'red': red,
        'green': green,
        'blue': blue,
        'thickness': thickness,
    }
    current_surprise = calculate_surprise(observed_glasses_params)

    # サプライズ値をグラフ上に表示
    ax.text(0.02, 0.98, f'Surprise: {current_surprise:.2f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    plt.show()

# --- ウィジェット設定 ---
shape_options = ['Circle', 'Square', 'Triangle']

# レイアウトを調整し、スライダーが見えるように変更
slider_layout = Layout(width='250px')
# Dropdownウィジェットの幅を広げる
dropdown_layout = Layout(width='180px')
label_layout = Layout(width='100px')

# UIスタイリング
button_style_default = ''
button_style_selected = 'info'

toggle_layout = Layout(width='auto', display='flex', flex_flow='row', justify_content='flex-start',
                       margin='0 0 10px 0', border='1px solid #B0BEC5', border_radius='5px',
                       padding='5px',
                       gap='5px'
)
button_layout = Layout(width='auto', flex_grow='0', padding='2px 10px', border='1px solid #78909C', border_radius='3px',
                       background_color='#CFD8DC',
                       color='#263238'
)
selected_button_layout = Layout(background_color='#B3E5FC',
                                border_color='#03A9F4',
                                color='#0D47A1'
)


# ヘルパー関数：スライダーとドロップダウンをHBoxにまとめる
def create_control_row(description, control_widget, layout=None):
    return HBox([widgets.Label(value=description, layout=label_layout), control_widget], layout=layout)

# 顔の高さスライダーを削除
def create_shape_controls_face(shape_options, current_shape, current_width_ratio):
    return HBox([
        Dropdown(options=shape_options, value=current_shape, description='形状:', layout=dropdown_layout),
        FloatSlider(value=current_width_ratio, min=0.5, max=2.0, step=0.05, description='幅比率:', layout=slider_layout, continuous_update=True),
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))

# 目の高さラベルを「目の大きさ」に変更
def create_shape_controls_eye(shape_options, current_shape, current_size, current_width_ratio, current_angle):
    return HBox([
        Dropdown(options=shape_options, value=current_shape, description='形状:', layout=dropdown_layout),
        FloatSlider(value=current_size, min=0.1, max=5.0, step=0.05, description='目の大きさ:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=current_width_ratio, min=0.5, max=2.0, step=0.05, description='幅比率:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=current_angle, min=-90, max=90, step=1, description='角度:', layout=slider_layout, continuous_update=True)
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))

# 鼻の高さラベルを「鼻の大きさ」に変更、角度を削除
def create_shape_controls_nose(shape_options, current_shape, current_size, current_width_ratio):
    return HBox([
        Dropdown(options=shape_options, value=current_shape, description='形状:', layout=dropdown_layout),
        FloatSlider(value=current_size, min=0.1, max=5.0, step=0.05, description='鼻の大きさ:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=current_width_ratio, min=0.5, max=2.0, step=0.05, description='幅比率:', layout=slider_layout, continuous_update=True),
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))

# メガネの高さラベルを「メガネの大きさ」に変更
def create_shape_controls_glasses(shape_options, current_shape, current_size, current_width_ratio, current_angle, description_prefix):
    return HBox([
        Dropdown(options=shape_options, value=current_shape, description=f'{description_prefix}形状:', layout=dropdown_layout),
        FloatSlider(value=current_size, min=0.5, max=2.0, step=0.1, description='大きさ:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=current_width_ratio, min=0.5, max=2.0, step=0.05, description='幅比率:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=current_angle, min=-180, max=180, step=1, description='角度:', layout=slider_layout, continuous_update=True)
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))


# 他のパーツの汎用コントロール
# 口のラベルを「大きさ」に変更
def create_shape_controls_general(shape_options, current_shape, current_size, current_width_ratio, current_angle):
    return HBox([
        Dropdown(options=shape_options, value=current_shape, description='形状:', layout=dropdown_layout),
        FloatSlider(value=current_size, min=0.1, max=5.0, step=0.05, description='大きさ:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=current_width_ratio, min=0.5, max=2.0, step=0.05, description='幅比率:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=current_angle, min=-180, max=180, step=1, description='角度:', layout=slider_layout, continuous_update=True)
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))


# --- 各パーツのパラメータグループ ---

# 顔パーツ
face_params = VBox([
    create_shape_controls_face(shape_options, 'Circle', 1.0),
])

# 眉毛 (眉毛の位置もここで調整)
brow_params = VBox([
    HBox([
        FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description='長さ:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=-90, max=90, step=1, description='角度:', layout=slider_layout, continuous_update=True),
        IntSlider(value=2, min=1, max=5, step=1, description='太さ:', layout=slider_layout, continuous_update=True)
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0')),
    HBox([ # 眉毛の共通オフセット
        FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.1, description='眉X:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.1, description='眉Y:', layout=slider_layout, continuous_update=True)
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))
])

# 目
eye_params = VBox([
    create_shape_controls_eye(shape_options, 'Circle', 0.3, 1.0, 0.0),
    HBox([ # 目の共通オフセット
        FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.1, description='目X:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.1, description='目Y:', layout=slider_layout, continuous_update=True)
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))
])

# 鼻用
nose_params = VBox([
    create_shape_controls_nose(shape_options, 'Triangle', 0.5, 1.0),
    HBox([ # 鼻のYオフセット
        FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.1, description='鼻Y:', layout=slider_layout, continuous_update=True),
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))
])

# 口用
mouth_params = VBox([
    create_shape_controls_general(shape_options, 'Circle', 0.7, 1.0, 0.0),
    HBox([ # 口のYオフセット
        FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.1, description='口Y:', layout=slider_layout, continuous_update=True),
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))
])

# 耳用
ear_params = VBox([
    HBox([
        FloatSlider(value=1.0, min=0.1, max=3.0, step=0.1, description='外側への膨らみ:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=1.5, min=0.5, max=3.0, step=0.1, description='上下幅:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=-2.5, max=2.5, step=0.1, description='上下接続位置:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=-45, max=45, step=1, description='傾き:', layout=slider_layout, continuous_update=True)
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))
])

# レンズ (形状と大きさ、色、透明度)
lens_params = VBox([
    create_shape_controls_glasses(shape_options, 'Circle', 1.0, 1.0, 0.0, '左レンズ'),
    create_shape_controls_glasses(shape_options, 'Circle', 1.0, 1.0, 0.0, '右レンズ'),
    HBox([ # レンズの色と透明度
        FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='レンズ赤:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='レンズ緑:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='レンズ青:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01, description='透明度:', layout=slider_layout, continuous_update=True) # 初期値を0.5に設定
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))
])

# フレーム (色と太さ)
frame_params = VBox([
    HBox([
        FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='フレーム赤:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='フレーム緑:', layout=slider_layout, continuous_update=True),
        FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='フレーム青:', layout=slider_layout, continuous_update=True)
    ], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0')),
    HBox([IntSlider(value=2, min=1, max=10, step=1, description='太さ:', layout=slider_layout, continuous_update=True)], layout=Layout(justify_content='space-around', flex_wrap='wrap', margin='0 0 5px 0'))
])


# --- 全てのパーツパラメータを格納する辞書 ---
all_part_params = {
    'face': face_params,
    'brow': brow_params,
    'eye': eye_params,
    'nose': nose_params,
    'mouth': mouth_params,
    'ear': ear_params,
    'lens': lens_params,
    'frame': frame_params,
}

# 編集パーツ選択 ToggleButtons のオプション
face_part_options_tuple = [
    ("顔", 'face'),
    ("眉毛", 'brow'),
    ("目", 'eye'),
    ("鼻", 'nose'),
    ("口", 'mouth'),
    ("耳", 'ear')
]
glasses_part_options_tuple = [
    ("レンズ", 'lens'),
    ("フレーム", 'frame')
]

# ToggleButtons ウィジェット
selected_part_toggle_face = ToggleButtons(
    options=face_part_options_tuple,
    value='face',
    button_style='',
    tooltips=[desc for desc, _ in face_part_options_tuple],
    layout=toggle_layout,
)

selected_part_toggle_glasses = ToggleButtons(
    options=glasses_part_options_tuple,
    value='lens',
    button_style='',
    tooltips=[desc for desc, _ in glasses_part_options_tuple],
    layout=toggle_layout,
)

# 選択パーツを統合して管理するダミーのセレクタ（内部で値を持つだけ）
selected_part_unified = widgets.Text(value='face', description='Current selection:', disabled=True, layout=Layout(display='none'))

# ToggleButtons の選択が変更されたときに、unifiedセレクタの値を更新する
# また、他の ToggleButtons の選択を解除する
def on_toggle_change_face(change):
    if change.new is not None:
        selected_part_unified.value = change.new
        selected_part_toggle_glasses.value = None

def on_toggle_change_glasses(change):
    if change.new is not None:
        selected_part_unified.value = change.new
        selected_part_toggle_face.value = None

selected_part_toggle_face.observe(on_toggle_change_face, names='value')
selected_part_toggle_glasses.observe(on_toggle_change_glasses, names='value')


# unifiedセレクタの値が変更されたときに、表示するパラメータセクションを切り替える
current_param_section = VBox([all_part_params['face']])

def on_unified_selection_change(change):
    part_key = change.new
    if part_key in all_part_params:
        current_param_section.children = [all_part_params[part_key]]
    else:
        current_param_section.children = []

selected_part_unified.observe(on_unified_selection_change, names='value')


# 各パーツのコントロールを格納する辞書 (更新)
all_controls = {
    'face_shape': face_params.children[0].children[0],
    'face_size': widgets.FloatSlider(value=3.0, min=1.0, max=5.0, step=0.1, description='顔のサイズ:', layout=slider_layout),
    'face_width_ratio': face_params.children[0].children[1],

    'brow_length': brow_params.children[0].children[0],
    'brow_angle': brow_params.children[0].children[1],
    'brow_thickness': brow_params.children[0].children[2],
    # 左右共通オフセット
    'brow_offset_x': brow_params.children[1].children[0],
    'brow_offset_y': brow_params.children[1].children[1],

    'eye_shape': eye_params.children[0].children[0],
    'eye_size': eye_params.children[0].children[1],
    'eye_width_ratio': eye_params.children[0].children[2],
    'eye_angle': eye_params.children[0].children[3],
    # 左右共通オフセット
    'eye_offset_x': eye_params.children[1].children[0],
    'eye_offset_y': eye_params.children[1].children[1],

    'nose_shape': nose_params.children[0].children[0],
    'nose_size': nose_params.children[0].children[1],
    'nose_width_ratio': nose_params.children[0].children[2],
    'nose_offset_y': nose_params.children[1].children[0],

    'mouth_shape': mouth_params.children[0].children[0],
    'mouth_size': mouth_params.children[0].children[1],
    'mouth_width_ratio': mouth_params.children[0].children[2],
    'mouth_angle': mouth_params.children[0].children[3],
    'mouth_offset_y': mouth_params.children[1].children[0],

    'ear_control_distance': ear_params.children[0].children[0],
    'ear_span': ear_params.children[0].children[1],
    'ear_y_attach_offset': ear_params.children[0].children[2],
    'ear_rotation': ear_params.children[0].children[3],

    'left_shape': lens_params.children[0].children[0],
    'left_size': lens_params.children[0].children[1],
    'left_width_ratio': lens_params.children[0].children[2],
    'left_angle': lens_params.children[0].children[3],
    'right_shape': lens_params.children[1].children[0],
    'right_size': lens_params.children[1].children[1],
    'right_width_ratio': lens_params.children[1].children[2],
    'right_angle': lens_params.children[1].children[3],

    'lens_red': lens_params.children[2].children[0],
    'lens_green': lens_params.children[2].children[1],
    'lens_blue': lens_params.children[2].children[2],
    'lens_alpha': lens_params.children[2].children[3],

    'red': frame_params.children[0].children[0],
    'green': frame_params.children[0].children[1],
    'blue': frame_params.children[0].children[2],
    'thickness': frame_params.children[1].children[0]
}

out = interactive_output(draw_face_and_glasses, {
    'selected_part': selected_part_unified,
    **all_controls
})

# --- 新しいUI要素: サプライズ目標スライダーとボタン ---
surprise_target_slider = FloatSlider(
    value=0.5, # 初期値は中程度のサプライズ
    min=0.0,
    max=1.0,
    step=0.01,
    description='目標サプライズレベル:',
    continuous_update=True,
    layout=slider_layout
)

generate_random_button = Button(
    description='ランダムデザイン生成',
    button_style='success',
    layout=Layout(width='auto', margin='0 0 10px 0')
)

# ボタンクリックイベントハンドラ
def on_generate_button_clicked(b):
    target_surprise = surprise_target_slider.value
    generate_and_set_random_glasses_design(target_surprise)

generate_random_button.on_click(on_generate_button_clicked)


# UIの組み立て
ui = VBox([
    widgets.Label(value="顔パーツの編集", layout=Layout(width='auto', font_weight='bold', color='#2E7D32', margin='0 0 5px 10px')),
    selected_part_toggle_face,

    widgets.Label(value="メガネパーツの編集", layout=Layout(width='auto', font_weight='bold', color='#2E7D32', margin='10px 0 5px 10px')),
    selected_part_toggle_glasses,

    current_param_section,

    widgets.Label(value="サプライズによるデザイン生成", layout=Layout(width='auto', font_weight='bold', color='#D84315', margin='10px 0 5px 10px')),
    HBox([surprise_target_slider, generate_random_button], layout=Layout(justify_content='flex-start', align_items='center', gap='20px')), # レイアウト調整

    out
], layout=Layout(border='3px solid #3F51B5', padding='15px', margin='10px', background_color='#E8EAF6', align_items='flex-start'))

display(ui)