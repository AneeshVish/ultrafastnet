"""
Physics-Informed Neural Network for 2D Flow Around a Sphere (Circle)
====================================================================
This code replaces the plate obstacle with a sphere (circle) and
solves the steady Navier-Stokes equations using a PINN. It includes
comprehensive domain setup, training, and visualization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os

# =========================
# DOMAIN AND GEOMETRY SETUP
# =========================

X_MIN, X_MAX = 0.0, 4.0
Y_MIN, Y_MAX = 0.0, 2.0

SPHERE_CENTER = (1.0, 1.0)
SPHERE_RADIUS = 0.1  # You can adjust the radius

def is_outside_sphere(x, y, center=SPHERE_CENTER, radius=SPHERE_RADIUS):
    cx, cy = center
    return (x - cx)**2 + (y - cy)**2 > radius**2

def sample_sphere(n=400, center=SPHERE_CENTER, radius=SPHERE_RADIUS):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def generate_fluid_points(n_points=8000):
    xs = np.random.uniform(X_MIN, X_MAX, n_points*3)
    ys = np.random.uniform(Y_MIN, Y_MAX, n_points*3)
    points = np.column_stack([xs, ys])
    mask = np.array([is_outside_sphere(x, y) for x, y in points])
    fluid_points = points[mask][:n_points]
    # Add extra points near the sphere for better resolution
    cx, cy = SPHERE_CENTER
    r = SPHERE_RADIUS
    ring_theta = np.random.uniform(0, 2*np.pi, n_points//4)
    ring_r = np.random.uniform(r+0.01, r+0.05, n_points//4)
    ring_x = cx + ring_r * np.cos(ring_theta)
    ring_y = cy + ring_r * np.sin(ring_theta)
    ring_points = np.column_stack([ring_x, ring_y])
    mask_ring = np.array([is_outside_sphere(x, y) for x, y in ring_points])
    fluid_points = np.vstack([fluid_points, ring_points[mask_ring]])
    return fluid_points[:n_points]

def sample_inlet(n=200):
    y = np.linspace(Y_MIN, Y_MAX, n)
    x = np.full_like(y, X_MIN)
    return np.column_stack([x, y])

def sample_outlet(n=200):
    y = np.linspace(Y_MIN, Y_MAX, n)
    x = np.full_like(y, X_MAX)
    return np.column_stack([x, y])

def sample_wall(y_val, n=200):
    x = np.linspace(X_MIN, X_MAX, n)
    y = np.full_like(x, y_val)
    return np.column_stack([x, y])

# =========================
# PINN ARCHITECTURE
# =========================

class PINN(nn.Module):
    def __init__(self, layers=[2, 80, 80, 80, 80, 80, 3], use_residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    def forward(self, x):
        x_norm = 2.0 * (x - torch.tensor([X_MIN, Y_MIN], device=x.device)) / torch.tensor([X_MAX-X_MIN, Y_MAX-Y_MIN], device=x.device) - 1.0
        u = x_norm
        for i, layer in enumerate(self.layers[:-1]):
            u_new = torch.tanh(layer(u))
            if self.use_residual and i > 0 and i < len(self.layers)-2:
                if u.shape[-1] == u_new.shape[-1]:
                    u = u_new + u
                else:
                    u = u_new
            else:
                u = u_new
        output = self.layers[-1](u)
        return output

# =========================
# PHYSICS LOSS FUNCTIONS
# =========================

def navier_stokes_residuals(model, xys, rho=1.225, nu=0.02):
    xys.requires_grad_(True)
    u_v_p = model(xys)
    u = u_v_p[:, 0:1]
    v = u_v_p[:, 1:2]
    p = u_v_p[:, 2:3]
    def grad(outputs, inputs, create_graph=True):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=create_graph, retain_graph=True, only_inputs=True)[0]
    u_xy = grad(u, xys)
    v_xy = grad(v, xys)
    p_xy = grad(p, xys)
    u_x = u_xy[:, 0:1]
    u_y = u_xy[:, 1:2]
    v_x = v_xy[:, 0:1]
    v_y = v_xy[:, 1:2]
    p_x = p_xy[:, 0:1]
    p_y = p_xy[:, 1:2]
    u_xx = grad(u_x, xys)[:, 0:1]
    u_yy = grad(u_y, xys)[:, 1:2]
    v_xx = grad(v_x, xys)[:, 0:1]
    v_yy = grad(v_y, xys)[:, 1:2]
    continuity = u_x + v_y
    momentum_u = u * u_x + v * u_y + (1/rho) * p_x - nu * (u_xx + u_yy)
    momentum_v = u * v_x + v * v_y + (1/rho) * p_y - nu * (v_xx + v_yy)
    return continuity, momentum_u, momentum_v

def physics_loss(model, fluid_points):
    pts = torch.tensor(fluid_points, dtype=torch.float32, device=next(model.parameters()).device)
    continuity, mom_u, mom_v = navier_stokes_residuals(model, pts)
    loss_continuity = torch.mean(continuity**2)
    loss_momentum_u = torch.mean(mom_u**2)
    loss_momentum_v = torch.mean(mom_v**2)
    total_loss = loss_continuity + loss_momentum_u + loss_momentum_v
    return total_loss

def boundary_loss(model, points, values, lambda_bc=10.0):
    pts = torch.tensor(points, dtype=torch.float32, device=next(model.parameters()).device)
    vals = torch.tensor(values, dtype=torch.float32, device=next(model.parameters()).device)
    prediction = model(pts)
    bc_loss = torch.mean((prediction[:, :2] - vals)**2)
    return lambda_bc * bc_loss

def outlet_neumann_loss(model, outlet_points, lambda_outlet=1.0):
    pts = torch.tensor(outlet_points, dtype=torch.float32, device=next(model.parameters()).device)
    pts.requires_grad_(True)
    pred = model(pts)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    grads_u = torch.autograd.grad(u, pts, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    grads_v = torch.autograd.grad(v, pts, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    du_dx = grads_u[:, 0:1]
    dv_dx = grads_v[:, 0:1]
    outlet_loss = torch.mean(du_dx**2) + torch.mean(dv_dx**2)
    return lambda_outlet * outlet_loss

def prepare_bc_values(boundary_name, points):
    if boundary_name == 'inlet':
        y_coords = points[:, 1]
        u_inlet = 4.0 * 1.0 * (y_coords - Y_MIN) * (Y_MAX - y_coords) / ((Y_MAX - Y_MIN)**2)
        v_inlet = np.zeros_like(u_inlet)
        values = np.column_stack([u_inlet, v_inlet])
    else:
        values = np.zeros((len(points), 2))
    return values

# =========================
# TRAINING AND VISUALIZATION
# =========================

def visualize_domain_setup():
    fluid_points = generate_fluid_points(5000)
    inlet_points = sample_inlet(200)
    outlet_points = sample_outlet(200)
    bottom_points = sample_wall(Y_MIN, 200)
    top_points = sample_wall(Y_MAX, 200)
    sphere_points = sample_sphere(400)
    plt.figure(figsize=(12, 6), dpi=150)
    plt.scatter(fluid_points[:,0], fluid_points[:,1], s=2, c="#85C1E9", label="Fluid Domain", alpha=0.6)
    plt.scatter(inlet_points[:,0], inlet_points[:,1], s=8, c="#FF6666", label="Inlet", alpha=0.8)
    plt.scatter(outlet_points[:,0], outlet_points[:,1], s=8, c="#FFD966", label="Outlet", alpha=0.8)
    plt.scatter(bottom_points[:,0], bottom_points[:,1], s=8, c="#82E0AA", label="Wall (bottom)", alpha=0.8)
    plt.scatter(top_points[:,0], top_points[:,1], s=8, c="#82E0AA", label="Wall (top)", alpha=0.8)
    plt.scatter(sphere_points[:,0], sphere_points[:,1], s=12, c="#E74C3C", label="Sphere", alpha=0.9)
    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)
    plt.gca().set_aspect('equal')
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper right')
    plt.title("Domain Setup: Fluid Flow Around Sphere", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def train_pinn(total_epochs=20000, checkpoint_interval=2000):
    print("="*70)
    print("PINN TRAINING FOR FLUID FLOW AROUND A SPHERE - 20,000 EPOCHS")
    print("="*70)
    training_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    fluid_points = generate_fluid_points(8000)
    inlet_points = sample_inlet(200)
    outlet_points = sample_outlet(200)
    bottom_points = sample_wall(Y_MIN, 200)
    top_points = sample_wall(Y_MAX, 200)
    sphere_points = sample_sphere(400)
    print(f"Fluid points: {len(fluid_points):,}")
    print(f"Boundary points: {len(inlet_points) + len(outlet_points) + len(bottom_points) + len(top_points) + len(sphere_points):,}")
    model = PINN(layers=[2, 80, 80, 80, 80, 80, 3], use_residual=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    bc_data = [
        (inlet_points, prepare_bc_values('inlet', inlet_points)),
        (bottom_points, prepare_bc_values('wall', bottom_points)), 
        (top_points, prepare_bc_values('wall', top_points)),
        (sphere_points, np.zeros((len(sphere_points), 2)))
    ]
    lr_initial = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    loss_history = []
    print(f"\nStarting training for {total_epochs:,} epochs...")
    print("="*70)
    epoch_times = []
    for epoch in range(total_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        loss_pde = physics_loss(model, fluid_points)
        loss_bc_total = 0
        for points, values in bc_data:
            loss_bc_total += boundary_loss(model, points, values, lambda_bc=20.0)
        loss_outlet = outlet_neumann_loss(model, outlet_points, lambda_outlet=5.0)
        total_loss = loss_pde + loss_bc_total + loss_outlet
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(total_loss.item())
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        if epoch % 500 == 0 or epoch == total_epochs-1:
            avg_epoch_time = np.mean(epoch_times[-100:]) if len(epoch_times) >= 100 else np.mean(epoch_times)
            remaining_epochs = total_epochs - epoch - 1
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_hours = eta_seconds / 3600
            print(f"Epoch {epoch:5d}/{total_epochs}: Total={total_loss.item():.4e} | PDE={loss_pde.item():.3e} | BC={loss_bc_total.item():.3e} | Outlet={loss_outlet.item():.3e} | ETA: {eta_hours:.2f}h")
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1:05d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_history': loss_history,
                'total_loss': total_loss.item()
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    training_time = time.time() - training_start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = training_time % 60
    print("="*70)
    print(f"Training completed!")
    print(f"Training time: {hours:02d}:{minutes:02d}:{seconds:05.2f} (hh:mm:ss)")
    print(f"Average time per epoch: {training_time/total_epochs:.3f} seconds")
    print(f"Final loss: {loss_history[-1]:.4e}")
    torch.save(model.state_dict(), "final_model_20k_epochs_sphere.pth")
    return model, loss_history, training_time

def create_visualization(model):
    print("Generating visualization...")
    viz_start_time = time.time()
    n_x, n_y = 1000, 400
    x = np.linspace(X_MIN, X_MAX, n_x)
    y = np.linspace(Y_MIN, Y_MAX, n_y)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    mask = np.array([is_outside_sphere(px, py) for px, py in grid_points])
    domain_points = grid_points[mask]
    device = next(model.parameters()).device
    model.eval()
    batch_size = 10000
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(domain_points), batch_size):
            batch = domain_points[i:i+batch_size]
            pts_torch = torch.tensor(batch, dtype=torch.float32, device=device)
            preds = model(pts_torch).cpu().numpy()
            all_predictions.append(preds)
    predictions = np.vstack(all_predictions)
    u_pred = predictions[:, 0]
    v_pred = predictions[:, 1]
    p_pred = predictions[:, 2]
    vel_mag = np.sqrt(u_pred**2 + v_pred**2)
    u_grid = np.full(xx.shape, np.nan)
    v_grid = np.full(xx.shape, np.nan)
    p_grid = np.full(xx.shape, np.nan)
    mag_grid = np.full(xx.shape, np.nan)
    u_grid.ravel()[mask] = u_pred
    v_grid.ravel()[mask] = v_pred
    p_grid.ravel()[mask] = p_pred
    mag_grid.ravel()[mask] = vel_mag
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('PINN (20,000 epochs): Fluid Flow Around Sphere', fontsize=18, fontweight='bold')
    levels_u = np.linspace(np.nanmin(u_grid), np.nanmax(u_grid), 50)
    im1 = axes[0,0].contourf(xx, yy, u_grid, levels=levels_u, cmap='RdBu_r', extend='both')
    axes[0,0].set_title('u velocity [m/s]', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('x [m]')
    axes[0,0].set_ylabel('y [m]')
    plt.colorbar(im1, ax=axes[0,0])
    levels_v = np.linspace(np.nanmin(v_grid), np.nanmax(v_grid), 50)
    im2 = axes[0,1].contourf(xx, yy, v_grid, levels=levels_v, cmap='RdBu_r', extend='both')
    axes[0,1].set_title('v velocity [m/s]', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('x [m]')
    axes[0,1].set_ylabel('y [m]')
    plt.colorbar(im2, ax=axes[0,1])
    levels_mag = np.linspace(0, np.nanmax(mag_grid), 50)
    im3 = axes[0,2].contourf(xx, yy, mag_grid, levels=levels_mag, cmap='viridis', extend='max')
    axes[0,2].set_title('Velocity Magnitude [m/s]', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('x [m]')
    axes[0,2].set_ylabel('y [m]')
    plt.colorbar(im3, ax=axes[0,2])
    levels_p = np.linspace(np.nanmin(p_grid), np.nanmax(p_grid), 50)
    im4 = axes[1,0].contourf(xx, yy, p_grid, levels=levels_p, cmap='coolwarm', extend='both')
    axes[1,0].set_title('Pressure [Pa]', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('x [m]')
    axes[1,0].set_ylabel('y [m]')
    plt.colorbar(im4, ax=axes[1,0])
    skip = 25
    axes[1,1].streamplot(xx[::skip, ::skip], yy[::skip, ::skip], u_grid[::skip, ::skip], v_grid[::skip, ::skip], density=2, color='black', linewidth=1.2)
    axes[1,1].set_title('Velocity Streamlines', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('x [m]')
    axes[1,1].set_ylabel('y [m]')
    sphere_patch = plt.Circle(SPHERE_CENTER, SPHERE_RADIUS, facecolor='red', edgecolor='black', linewidth=2, alpha=0.8)
    axes[1,1].add_patch(sphere_patch)
    axes[1,1].set_xlim(X_MIN, X_MAX)
    axes[1,1].set_ylim(Y_MIN, Y_MAX)
    axes[1,1].set_aspect('equal')
    axes[1,1].grid(True, alpha=0.3)
    im5 = axes[1,2].contourf(xx, yy, mag_grid, levels=30, cmap='viridis', alpha=0.8, extend='max')
    axes[1,2].streamplot(xx[::skip, ::skip], yy[::skip, ::skip], u_grid[::skip, ::skip], v_grid[::skip, ::skip], density=1.8, color='white', linewidth=1)
    axes[1,2].set_title('Combined: Velocity + Streamlines', fontsize=14, fontweight='bold')
    axes[1,2].set_xlabel('x [m]')
    axes[1,2].set_ylabel('y [m]')
    plt.colorbar(im5, ax=axes[1,2])
    sphere_patch2 = plt.Circle(SPHERE_CENTER, SPHERE_RADIUS, facecolor='red', edgecolor='black', linewidth=2, alpha=0.9)
    axes[1,2].add_patch(sphere_patch2)
    axes[1,2].set_xlim(X_MIN, X_MAX)
    axes[1,2].set_ylim(Y_MIN, Y_MAX)
    axes[1,2].set_aspect('equal')
    axes[1,2].grid(True, alpha=0.3)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    viz_time = time.time() - viz_start_time
    print(f"Visualization completed in {viz_time:.2f} seconds")
    return viz_time

def plot_training_history(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Loss (20,000 epochs)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    global_start = time.time()
    print("="*80)
    print("COMPLETE PINN FOR 2D FLOW AROUND A SPHERE - 20,000 EPOCHS")
    print("="*80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("="*80)
    print("\n1. Visualizing domain setup...")
    visualize_domain_setup()
    print("\n2. Training PINN model...")
    model, loss_history, training_time = train_pinn(total_epochs=20000, checkpoint_interval=2000)
    print("\n3. Creating fluid flow visualization...")
    viz_time = create_visualization(model)
    print("\n4. Plotting training history...")
    plot_training_history(loss_history)
    total_time = time.time() - global_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Total wall-clock time: {hours:02d}:{minutes:02d}:{seconds:05.2f} (hh:mm:ss)")
    print(f"Training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    print(f"Visualization time: {viz_time:.2f} seconds")
    print(f"Final training loss: {loss_history[-1]:.6e}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    device = next(model.parameters()).device
    print(f"\nPerformance Summary:")
    print(f"Device used: {device}")
    print("\n✅ PINN for SPHERE completed successfully.")
    print("="*80)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    main()
