"""Training utilities (seeding, metric logging) and the evaluation-video overlays: value heatmaps, subgoals, plans."""

import random

import numpy as np
import torch
import cv2


# agents that hand subgoals to a low-level policy; their subgoals are drawn on evaluation frames
SUBGOAL_AGENTS = ('telawm', 'hiql', 'lacwm', 'acwm', 'sd')


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set seed to {seed}")


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def visualize_frame(env, goal_info, vis_info):
    """
    Visualize the current frame of the environment, stacked below the goal frame (and an image-space subgoal, if any).
    """
    
    frame = env.render(vis_info)
    goal_frame = goal_info['goal_rendered']
    
    if vis_info.get("subgoal_image") is not None:
        # image-space subgoal (e.g. a decoded latent subgoal): goal / subgoal / frame, top to bottom
        subgoal_frame = vis_info["subgoal_image"].reshape(-1, *vis_info["subgoal_image"].shape[-3:])[0]
        subgoal_frame = subgoal_frame.detach().float().clamp(0, 255).cpu().numpy().astype(np.uint8)
        subgoal_frame = cv2.resize(subgoal_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        frame = np.concatenate([goal_frame, subgoal_frame, frame], axis=0)

    else:
        frame = np.concatenate([goal_frame, frame], axis=0)

    return frame


def extract_maze_type(task_name):
    """Extract maze type from task name (e.g., 'locomaze-large-navigate-v0' -> 'large')."""
    for maze_type in ['arena', 'medium', 'large', 'giant', 'teleport']:
        if maze_type in task_name:
            return maze_type
    return 'medium'  # default


def get_vis_info(cfg, step, goal, obs, obs_mean, obs_std, agent, act_info):
    """Collect what OGBenchWrapper.render needs to draw value maps, subgoals and plans on an evaluation frame."""
    # image / latent observations have no world coordinates for the overlays below
    if cfg.obs in ('rgb', 'latent'):
        return {"subgoal_image": act_info.get("subgoal_image")}

    is_maze, is_soccer = 'maze' in cfg.task, 'soccer' in cfg.task
    if not (is_maze or is_soccer or 'cube' in cfg.task or 'scene' in cfg.task):
        return {}

    vis_info = dict(obs=obs, obs_mean=obs_mean, obs_std=obs_std, goal=goal, value_fn=agent.vf, device=agent.device)
    if is_maze or is_soccer:
        vis_info.update(maze_type=extract_maze_type(cfg.task), task=cfg.task)
        if is_maze:  # soccer: compute_value_map is set in OGBenchWrapper.render when the ball xy changes
            vis_info["compute_value_map"] = step <= cfg.video_frame_skip
        if cfg.agent == 'tdmpc':
            vis_info.update(dynamics_fn=agent._forward_dynamics, planned_actions=act_info["planned_actions"])
    if cfg.agent in SUBGOAL_AGENTS:
        vis_info.update(subgoal=act_info["subgoal"].squeeze(0), subgoal_samples=act_info.get("subgoal_samples"))

    return vis_info


def get_maze_bounds(maze_type):
    """
    Returns coordinate bounds for different maze types.
    Each cell in ogbench mazes is 4 units wide.
    The walkable area is (cols-2) x (rows-2) cells (excluding boundary walls).
    
    Returns:
        (x_min, x_max, y_min, y_max) tuple of coordinate bounds
    """
    maze_configs = {
        'arena': {'cols': 8, 'rows': 8},    # 8x8 square
        'medium': {'cols': 8, 'rows': 8},    # 8x8 square
        'large': {'cols': 12, 'rows': 9},    # 12x9 (wider than tall)
        'giant': {'cols': 16, 'rows': 12},   # 16x12 (wider than tall)
        'teleport': {'cols': 12, 'rows': 9},    # 12x9 (wider than tall)
    }
    
    config = maze_configs[maze_type]
    cols, rows = config['cols'], config['rows']
    
    cell_size = 4
    padding = 2
    
    # For medium: x_min=-2, x_max=22 (range=24=6*4)
    x_min = -padding
    x_max = (cols - 2) * cell_size - padding
    y_min = -padding
    y_max = (rows - 2) * cell_size - padding
    
    return x_min, x_max, y_min, y_max


def get_maze_pixel_transform(maze_type, width, height):
    """
    Computes transform parameters for mapping between maze and pixel coordinates
    while preserving aspect ratio. The maze is centered in the image.
    
    Returns:
        (scale, x_offset, y_offset, x_min, x_max, y_min, y_max)
        - scale: pixels per maze unit (uniform for both axes)
        - x_offset, y_offset: pixel offsets to center the maze
        - x_min, x_max, y_min, y_max: maze coordinate bounds
    """
    x_min, x_max, y_min, y_max = get_maze_bounds(maze_type)
    
    maze_width = x_max - x_min
    maze_height = y_max - y_min
    
    # Use uniform scaling (fit maze within image, preserving aspect ratio)
    scale_x = width / maze_width
    scale_y = height / maze_height
    scale = min(scale_x, scale_y)
    
    # Compute offsets to center the maze
    rendered_width = maze_width * scale
    rendered_height = maze_height * scale
    x_offset = (width - rendered_width) / 2
    y_offset = (height - rendered_height) / 2
    
    return scale, x_offset, y_offset, x_min, x_max, y_min, y_max


def get_value_text_position(maze_type):
    """
    Returns hardcoded pixel position (x, y) for displaying the value text on a wall.
    Positions are tuned for 200x200 images.
    """
    positions = {
        'arena': (140, 120),
        'medium': (140, 120),
        'large': (120, 105),
        'giant': (85, 68),
        'teleport': (37, 125),
    }
    return positions[maze_type]


def plot_state_value_on_frame(frame, info):
    """Draw the scalar value estimate V(s, goal) on a rendered frame (white text, top-left)."""
    obs = (info["obs"].unsqueeze(0) - info["obs_mean"]) / info["obs_std"]
    cur_value = info["value_fn"](obs.to(info["device"]), info["goal"].to(info["device"]))
    text = f"{cur_value.item():.3f}"
    frame = cv2.putText(frame, text, org=(6, 16), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return frame


def plot_value_on_maze(frame, value, info):
    """
    Blend a heatmap of V(s, goal) over the maze area of the frame, varying the agent xy over the pixel grid.

    The heatmap is recomputed when info["compute_value_map"] is set; otherwise the passed `value` grid is reused.
    Returns the frame and the [H, W] value grid.
    """
    maze_type = info.get("maze_type", "medium")
    H, W = frame.shape[:2]
    scale, x_off, y_off, x_min, x_max, y_min, y_max = get_maze_pixel_transform(maze_type, W, H)
    
    # create mask for pixels within the maze bounds
    px_grid, py_grid = np.meshgrid(np.arange(W), np.arange(H))
    x_coords = (px_grid - x_off) / scale + x_min
    y_coords = y_max - (py_grid - y_off) / scale
    maze_mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
    
    # maybe recompute value
    if info["compute_value_map"]:
        with torch.no_grad():
            value_fn = info["value_fn"]
            
            # create the grid coordinates (aspect-ratio preserving)
            px, py = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
            pxy = torch.stack((px, py), dim=-1)
            flat_pxy = pxy.view(-1, 2).float()
            
            # convert pixel to maze coords with uniform scaling
            x, y = pixel_to_maze_uniform(
                flat_pxy[:, 0], flat_pxy[:, 1],
                scale, x_off, y_off, x_min, y_max, H
            )
            xy_flat = torch.stack((x, y), dim=-1)

            # prepare inputs to value function (vary agent/ant xy in [:2])
            # maze: template=goal (agent near goal xy)
            # soccer: template=current obs so ball stays fixed; goal passed separately to vf
            if 'soccer' in info["task"]:
                unnormalized_template = info["obs"].detach().float().reshape(-1)
            else:
                unnormalized_template = (info["goal"] * info["obs_std"] + info["obs_mean"]).detach().float().reshape(-1)
            state_grid_flat = unnormalized_template.unsqueeze(0).repeat(len(xy_flat), 1).clone()
            state_grid_flat[:, :2] = xy_flat

            # allocate result tensor
            value = torch.empty(state_grid_flat.shape[0])

            # process in batches
            batch_size = 1024  # adjust based on memory constraints
            goal_batch = info["goal"].repeat(batch_size, 1)
            for i in range(0, state_grid_flat.shape[0], batch_size):
                unnormalized_state_batch = state_grid_flat[i:i+batch_size]
                state_batch = (unnormalized_state_batch - info["obs_mean"]) / info["obs_std"]
                value[i:i+batch_size] = value_fn(state_batch.to(info["device"]), goal_batch[:len(state_batch)].to(info["device"])).squeeze()

            # reshape back to grid
            value = value.view(H, W).numpy()
    
    # normalize heatmap data (0 to 255) only within maze region for proper scaling
    maze_values = value[maze_mask]
    if maze_values.size > 0 and np.ptp(maze_values) > 0:
        heatmap = np.uint8(255 * (value - np.min(maze_values)) / np.ptp(maze_values))
    else:
        heatmap = np.zeros_like(value, dtype=np.uint8)

    # apply a colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # blend the heatmap with the original image only within maze region
    alpha = 0.4
    blended = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
    
    # apply mask: use blended where inside maze, original frame where outside
    mask_3ch = np.stack([maze_mask] * 3, axis=-1)
    frame = np.where(mask_3ch, blended, frame)

    # write current value on image (positioned on a wall cell)
    obs = (info["obs"].unsqueeze(0) - info["obs_mean"]) / info["obs_std"]
    cur_value = info["value_fn"](obs.to(info["device"]), info["goal"].to(info["device"]))
    # text position and size are tuned for 200x200 frames
    px, py = get_value_text_position(maze_type)
    text_pos = (int(px * W / 200), int(py * H / 200))
    text_scale = 0.4 * min(W, H) / 200
    frame = cv2.putText(frame, f"{cur_value.item():.3f}", org=text_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, color=(0, 0, 0), thickness=max(1, round(min(W, H) / 200)), lineType=cv2.LINE_AA)

    return frame, value


def plot_plan_on_maze(frame, info):
    actions = info["planned_actions"].to(info["device"])
    dynamics_fn = info["dynamics_fn"]
    maze_type = info["maze_type"]
    
    colors = cold_to_warm_gradient(len(actions))
    
    obs = (info["obs"] - info["obs_mean"]) / info["obs_std"]
    obs = obs.to(info["device"])
    H, W = frame.shape[:2]
    with torch.no_grad():
        for i, action in enumerate(actions):
            frame = plot_state_xy_on_maze(frame, obs.cpu() * info["obs_std"] + info["obs_mean"], colors[i], maze_type=maze_type, width=W, height=H)
            obs = dynamics_fn(obs, action)
    
    return frame


def plot_state_xy_on_maze(frame, obs, color=(0, 0, 255), maze_type="medium", width=None, height=None, alpha: float = 1.0, radius: int = 2, xy_slice=slice(0, 2)):
    if width is None:
        width = frame.shape[1]
    if height is None:
        height = frame.shape[0]
    xy = obs[xy_slice]
    x = xy[0].item() if torch.is_tensor(xy[0]) else float(xy[0])
    y = xy[1].item() if torch.is_tensor(xy[1]) else float(xy[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        print(f"plot_state_xy_on_maze: skipping non-finite coordinates (x={x}, y={y})")
        return frame
    scale, x_off, y_off, x_min, x_max, y_min, y_max = get_maze_pixel_transform(maze_type, width, height)
    px, py = maze_to_pixel_uniform(x=x, y=y, scale=scale, x_off=x_off, y_off=y_off, x_min=x_min, y_max=y_max, height=height)
    center = (int(px), int(py))
    if alpha >= 1.0:
        frame = cv2.circle(frame, center, radius=radius, color=color, thickness=-1)
    else:
        overlay = frame.copy()
        overlay = cv2.circle(overlay, center, radius=radius, color=color, thickness=-1)
        frame = cv2.addWeighted(overlay, float(alpha), frame, float(1.0 - alpha), 0.0)
    return frame


def plot_states_xy_on_maze(frame, obs_batch, color=(0, 0, 255), maze_type="medium", alpha: float = 1.0, xy_slice=slice(0, 2), radius: int = 2):
    obs_flat = obs_batch.reshape(-1, obs_batch.shape[-1])
    H, W = frame.shape[:2]
    for obs in obs_flat:
        frame = plot_state_xy_on_maze(
            frame, obs, color=color, maze_type=maze_type, width=W, height=H,
            alpha=alpha, radius=radius, xy_slice=xy_slice,
        )
    return frame


def maze_to_pixel_uniform(x, y, scale, x_off, y_off, x_min, y_max, height):
    """
    Maps maze coordinates to pixel coordinates with uniform scaling (preserves aspect ratio).
    The maze is centered in the image.
    
    Parameters:
    x, y -- Maze coordinates
    scale -- Pixels per maze unit (uniform for both axes)
    x_off, y_off -- Pixel offsets to center the maze
    x_min -- Minimum x bound of maze
    y_max -- Maximum y bound of maze
    height -- Image height (for y-axis inversion)
    
    Returns:
    (px, py) -- Pixel coordinates
    """
    px = int((x - x_min) * scale + x_off)
    py = int((y_max - y) * scale + y_off)  # y is inverted
    return px, py


def pixel_to_maze_uniform(px, py, scale, x_off, y_off, x_min, y_max, height):
    """
    Maps pixel coordinates to maze coordinates with uniform scaling (preserves aspect ratio).
    
    Parameters:
    px, py -- Pixel coordinates (can be tensors)
    scale -- Pixels per maze unit (uniform for both axes)
    x_off, y_off -- Pixel offsets to center the maze
    x_min -- Minimum x bound of maze
    y_max -- Maximum y bound of maze
    height -- Image height (for y-axis inversion)
    
    Returns:
    (x, y) -- Maze coordinates
    """
    x = (px - x_off) / scale + x_min
    y = y_max - (py - y_off) / scale  # y is inverted
    return x, y


def cold_to_warm_gradient(n: int) -> np.ndarray:
    anchor_colors = np.array([
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 1.0, 1.0],  # Cyan
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.5, 0.0],  # Orange
        [1.0, 0.0, 0.0],  # Red
    ])

    anchor_steps = np.linspace(0, 1, len(anchor_colors))
    gradient_steps = np.linspace(0, 1, n)

    r = np.interp(gradient_steps, anchor_steps, anchor_colors[:, 0])
    g = np.interp(gradient_steps, anchor_steps, anchor_colors[:, 1])
    b = np.interp(gradient_steps, anchor_steps, anchor_colors[:, 2])

    rgb_gradient = np.stack([r, g, b], axis=1)  # shape: (n, 3)

    return (rgb_gradient * 255)