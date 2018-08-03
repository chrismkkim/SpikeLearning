function run_target(M,net_param,target_type)

N, p, g, b, dt, T = net_param;
N = Int(N);

t   = collect(dt:dt:T);

#---------- 1. periodic functions ----------#
if target_type == "periodic"
    u_periodic  = zeros(length(t),N);
    for j=1:N
        # A  = 0.5 + 1.5*rand();
        A  = 0.5 + 1.0*rand();
        T1 = 50. + 50.*rand();
        T2 = 10. + 40.*rand();
        t1 = T1*rand();
        t2 = T2*rand();

        u_periodic[:,j] = A*sin((t-t1)*(2*pi/T1)).*sin((t-t2)*(2*pi/T2));
    end

    return u_periodic
end

#---------- 2. rate network ----------#
if target_type == "ratemodel"
    u_rm   = zeros(length(t),N);
    g_rm = 5;
    M = g_rm*genw_sparse(N,0,1,p)/sqrt(N*p);
    b_rm = 1/4;

    # the mean of incoming synaptic connections to a neuron is zero
    for i = 1:N
      idx = abs(M[i,:]) .> 0;
      M[i,idx] = M[i,idx] - sum(M[i,idx])/sum(idx);
    end

    # random initial condition
    u_rm[1,:] = 0.2*rand(N)';

    function phi(x)
        x[x .< 0.] = 0.;
        y = sqrt(x)/pi;
        return y
    end

    for i = 1:length(t)-1
        u_rm[i+1,:] = (1-dt*b_rm)*u_rm[i,:] + (dt*b_rm*M*phi(u_rm[i,:]'))';
    end
    return u_rm
end

#---------- 3. Ornstein-Ulenbeck ----------#
if target_type == "ou"
    u_ou    = zeros(length(t),N);
    b_ou  = 1/20;
    mu    = 0.0;
    sig   = 0.3;

    for j = 1:N
        for i = 1:length(t)-1
            u_ou[i+1,j] = u_ou[i,j]+b_ou*(mu-u_ou[i,j])*dt + sig*sqrt(dt)*randn();
        end
    end
    return u_ou
end

end
