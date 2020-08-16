import taichi as ti
import numpy as np
import math
import taichi_three as t3

ti.init(ti.gpu)


# 
screen_res = (800,400)
screen_to_world_ratio = 1.0
bd = (20,15,10)
boundary = (bd[0]/screen_to_world_ratio,bd[1]/screen_to_world_ratio,bd[2]/screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size
# 
def round_up(f,s):
    return (math.floor(f * cell_recpr / s) + 1) * s

#网格数量、
grid_size = (round_up(boundary[0],1),round_up(boundary[1],1),round_up(boundary[2],1))

# 
dim = 3   # 维度
num_particles_x = 15
num_particles_y = 10
num_particles_z = 10
num_particles = num_particles_x * num_particles_y * num_particles_z 
max_num_particles_per_cell = 100
particle_radius = 0.15
particle_radius_in_world = particle_radius / screen_to_world_ratio
epsilon = 1e-5
max_num_neighbors = 100

# PBF param
h = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pdf_num_iters = 5
neighbor_radius = h * 1.05
corr_deltaQ_coeff = 0.3
corrK = 0.001


poly6_factor = 315.0 / 64.0 / np.pi
spiky_grad_factor = -45.0 / np.pi

# 三维空间数据
old_positions = ti.Vector.field(dim, dtype=ti.f32)
positions = ti.Vector.field(dim, dtype=ti.f32)
velocities = ti.Vector.field(dim, dtype=ti.f32)
time_delta = 1.0 / 20.0
grid_num_particles = ti.field(ti.i32)
grid2particles = ti.field(ti.i32)
particle_num_neighbors = ti.field(ti.i32)
particle_neighbors = ti.field(ti.i32)
lambdas = ti.field(ti.f32)
position_deltas = ti.Vector.field(dim, dtype=ti.f32)
#
board_states = ti.Vector.field(2,dtype=ti.f32)


# 
ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
grid_snode = ti.root.dense(ti.ijk,grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l,max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i,num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)

ti.root.dense(ti.i , num_particles).place(lambdas, position_deltas)
ti.root.place(board_states)

# taichi_three
scene = t3.SceneRT(screen_res)
radius = ti.field(ti.f32, num_particles)

scene.add_ball(positions, radius)
scene.set_light_dir([1, 1, -1])


#

@ti.func
def poly6_value(s,h):
    result = 0.0
    if 0 < s and s < h :
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result 

@ti.func
def spiky_grandient(r,h):
    result = ti.Vector([0.0,0.0,0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def compute_scorr(pos_ji):
    #Eq(13)
    x = poly6_value(pos_ji.norm(),h) / poly6_value(corr_deltaQ_coeff * h, h)

    x = x * x
    x = x * x
    return (-corrK) * x
 
@ti.func
def get_cell(pos):
    return (pos * cell_recpr).cast(int)

@ti.func
def is_in_grid(c):
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1] and 0 <= c[2] and c[2] < grid_size[2]

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0],boundary[1],boundary[2]]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <=bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()

    return p
    


@ti.kernel
def blit_buffers(f:ti.template(),t:ti.template()):
    for i in f:
        t[i] = f[i]

@ti.kernel
def move_board():
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 3.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b

@ti.kernel
def apply_gravity_within_boundary():
    for i in positions:
        g = ti.Vector([0.0,-9.8,0.0])
        pos,vel = positions[i],velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)

@ti.kernel
def confine_to_boundary():
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)

@ti.kernel
def find_particle_neighbors():
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1,2),(-1,2),(-1,2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i,nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.kernel
def compute_lambdas():
    #Eq(8) ~Eq(11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0,0.0,0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i,j]

            if p_j >= 0:
                pos_ji = pos_i - positions[p_j]
                grad_j = spiky_grandient(pos_ji,h)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                #Eq(2)
                density_constraint += poly6_value(pos_ji.norm(), h)

        #Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)

@ti.kernel
def compute_position_deltas():
    #Eq(12),Eq(14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0,0.0,0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]

            if p_j >= 0:
                lambda_j = lambdas[p_j]
                pos_ji = pos_i - positions[p_j]
                scorr_ij = compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_grandient(pos_ji, h)
        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i

@ti.kernel
def apply_position_deltas():
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def update_grid():
    for p_i in positions:
        cell = get_cell(positions[p_i])
        offs = grid_num_particles[cell].atomic_add(1)
        grid2particles[cell, offs] = p_i

@ti.kernel
def update_velocities():
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta

def run_pdf():
    blit_buffers(positions, old_positions)
    apply_gravity_within_boundary()

    grid_num_particles.fill(0)
    particle_neighbors.fill(-1)
    update_grid()
    find_particle_neighbors()
    for _ in range(pdf_num_iters):
        compute_lambdas()
        compute_position_deltas()
        apply_position_deltas()

    confine_to_boundary()
    update_velocities()
    #  vorticity/xsph 


#初始化球的位置、速度、半径
def init_particles():
    np_positions = np.zeros((num_particles,dim),dtype=np.float32)
    delta = h * 0.8
    num_x = num_particles_x
    num_y = num_particles_y
    num_z = num_particles // num_particles_x // num_particles_y
    assert num_x * num_y * num_z == num_particles 
    offs = np.array([0.0 , (boundary[1] * 0.1) , 0.0],dtype=np.float32)
    # offs = np.array([( - delta * num_x / screen_to_world_ratio) * 0.5,(boundary[1] * 0.02),( - delta * num_z / screen_to_world_ratio) * 0.5],dtype=np.float32)
    
    for i in range(num_particles):
        # print(i)
        # 
        np_positions[i] = np.array([i % num_x ,i // (num_x * num_z) ,(i - ((i // (num_x * num_z)) * (num_x * num_z))) // num_x ]) / screen_to_world_ratio * delta + offs
        #


    np_velocities = np.zeros((num_particles,dim),dtype=np.float32)

    @ti.kernel
    def init(p:ti.ext_arr(),v:ti.ext_arr()):
        for i in range(num_particles):
            radius[i] = particle_radius_in_world
            for c in ti.static(range(dim)):
                positions[i][c] = p[i,c]
                velocities[i][c] = v[i,c]

    @ti.kernel
    def init2():
        board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

    init(np_positions,np_velocities)
    init2()
 

init_particles()
gui = ti.GUI('PBF_3D', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)

    move_board()
    run_pdf()

    scene.camera.from_mouse(gui, dis=30)
    scene.render()
    
    gui.set_image(scene.img)
    
    gui.show()
    