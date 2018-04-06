function run_train(M,net_param,train_param,time_param,dt,utarg)

N        = length(M[1,:]);
w        = zeros(N,N);
extinput = zeros(N);
u        = zeros(N);

x, r, b = net_param[1:N], net_param[N+1:2*N], net_param[2*N+1];
stim, lambda, learn_every, nloop, target_type = train_param[1:N], train_param[N+1], train_param[N+2], train_param[N+3], train_param[N+4];
stim_on, stim_off, train_time = time_param[1], time_param[2], time_param[3];

# RHS of ODEs
dtheta(x_var,I_var) = 1 - cos(x_var) + I_var.*(1 + cos(x_var));
dr(r_var) = -b*r_var;

# set up correlation matrix
P = Dict{Int64,Array{Float64,2}}();
Px = Dict{Int64,BitArray{2}}();
for ni=1:Int(N)
    ni_prenum = sum(M[ni,:].!=0);
    ni_preind = M[ni,:].!=0;
    P[ni] = (1.0/lambda)*eye(ni_prenum);
    Px[ni] = ni_preind;
end

t = 0.;
for iloop = 1:Int(nloop)
    print("Loop $(iloop)...",'\n')
    for i=1:Int(train_time/dt)
          # External stimulus
          if i > Int(stim_on/dt) && i <= Int(stim_off/dt)
                extinput[:] = stim[:];
          else
                extinput[:] = zeros(N)[:];
          end

          # Update neuron phase
          k1 = dt*dtheta(x,      u+extinput);
          k2 = dt*dtheta(x+k1/2, u+extinput);
          k3 = dt*dtheta(x+k2/2, u+extinput);
          k4 = dt*dtheta(x+k3,   u+extinput);
          xnext  = x + (k1+2*k2+2*k3+k4)/6;

          # Update filtered spikes
          l1 = dt*dr(r);
          l2 = dt*dr(r+l1/2);
          l3 = dt*dr(r+l2/2);
          l4 = dt*dr(r+l3);

          # Spike detection.
          idx1 = xnext - x .> 0.;
          idx2 = xnext - x .> mod(pi - mod(x,2*pi),2*pi); # distance to next pi.
          idx  = idx1.*idx2;
          ind  = collect(1:N)[idx[:]]';

          # Update each neuron's spike
          if !isempty(ind)
              r[idx[:]] = r[idx[:]] + b;
          end

          # Update x, r, z, t.
          x = x + (k1+2*k2+2*k3+k4)/6;
          r = r + (l1+2*l2+2*l3+l4)/6;
          u = M*r;
          t = t + dt;

          # Train w
          if i > Int(stim_off/dt)+1 && i <= Int(train_time/dt) && mod(i, learn_every/dt) == 0
              for ni=1:Int(N)
                	# update inverse correlation matrix
                	k     = P[ni]*r[Px[ni][:]];
                	vPv   = r[Px[ni][:]].'*k;
                	den   = 1.0/(1.0 + vPv[1]);
                	P[ni] = P[ni] - k*(k.'*den);

                	# update recurrent weights
                  e  = M[ni,Px[ni][:]]*r[Px[ni][:]] - utarg[i,ni];
                	dw = -e[1]*k*den;
                	M[ni,Px[ni][:]] = M[ni,Px[ni][:]] + dw';
              end
          end
    end
end

return M

end
