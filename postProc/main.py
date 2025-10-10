dx = 0.1
dy = 1
dz = 1

beta_x = dy*dz/dx
beta_y = dx*dz/dy
beta_z = dy*dx/dz

print(f"{beta_x = }, {beta_y = }, {beta_z = }")

gamm_all = beta_x + beta_z + beta_y

gamma_x = beta_x / gamm_all
gamma_y = beta_y / gamm_all
gamma_z = beta_z / gamm_all

print(f"{gamma_x = }, {gamma_y = }, {gamma_z = }")
