function run_test(M,net_param,time_param,dt)

N = length(M[1,:]);
extinput = zeros(N);
u = zeros(N);

x, r, b, stim = net_param[1:N], net_param[N+1:2*N], net_param[2*N+1], net_param[2*N+2:3*N+1];
stim_on, stim_off, test_time = time_param[1], time_param[2], time_param[3];

#===== Run simulation =====#
dtheta(x_var,I_var) = 1. - cos(x_var) + I_var.*(1. + cos(x_var));
dr(r_var) = -b*r_var;

t = 0.;

utest = zeros(Int(test_time/dt),N);

for i=1:Int(test_time/dt)

      if i > Int(stim_on/dt) && i <= Int(stim_off/dt)
            extinput[:] = stim[:];
      else
            extinput[:] = zeros(N)[:];
      end

      # Define synaptic activity
      u = M*r;

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

      # Update each neuron's spikes
      if !isempty(ind)
          r[idx[:]] = r[idx[:]] + b;
      end

      # Update x, r
      x = x + (k1+2*k2+2*k3+k4)/6;
      r = r + (l1+2*l2+2*l3+l4)/6;
      u = M*r;
      t = t + dt;

      # save test data
      utest[i,:] = u[:];
end

return utest

end
