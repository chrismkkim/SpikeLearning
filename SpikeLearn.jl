using Distributions
using PyPlot

include("genw_sparse.jl")
include("run_target.jl")
include("run_train.jl")
include("run_test.jl")

# connectivity matrix
N = 200;
p = 0.3;
g = 1.0;

# neuron dynamics
b  = 1/2;

# training variables
train_duration = 100.;
nloop          = 10; #30
lambda         = 0.5; # 1, 10
learn_every    = 1.0;

# training param
stim_on         = 100.;
stim_off        = 110.;
train_time      = stim_off + train_duration;
test_time       = train_time + 100.;

dt = 0.1;

#---------- Choose target type ----------#
# target_type = "periodic";
target_type = "ratemodel";
# target_type = "ou";


#---------- Generate target trajectories ----------#
# generate M
M = g*genw_sparse(N,0,1,p)/sqrt(N*p);

# generate stim
stim = 2*(2*rand(N)-1);

# generate target patterns
net_param = [N, p, g, b, dt, test_time];
utarg = run_target(M,net_param,target_type);


#---------- Train recurrent connectivity ----------#
# Initialize variables
x0 = 2*pi*rand(N);
r0 = zeros(N);

# Run training
net_param   = [x0, r0, b];
train_param = [stim, lambda, learn_every, nloop, target_type];
time_param  = [stim_on, stim_off, train_time]
M = run_train(M,net_param,train_param,time_param,dt,utarg);


#---------- Test ----------#
# Initialize variables
x0 = 2*pi*rand(N);
r0 = zeros(N);
z0 = zeros(N);

net_param = [x0, r0, b, stim];
time_param = [stim_on, stim_off, test_time];
utest = run_test(M,net_param,time_param,dt);


#---------- Plot results ----------#
tvec = collect(dt: dt : test_time);

figure(figsize=(10,5))
nrow = 2;
ncol = 2;

for i = 1:nrow*ncol
    subplot(2,2,i)
    axvspan(10*stim_on,10*stim_off,alpha=0.5,color="dodgerblue")
    axvspan(10*stim_off,10*train_time,alpha=0.5,color="gray")

    ton = Int(stim_off/dt);
    toff = Int(train_time/dt);
    plot(tvec[ton:toff]*10,utarg[ton:toff,i],linewidth=3,color="black",label="target")
    plot(tvec*10,utest[:,i],linewidth=2,color="red",alpha=1,label="actual")

    xlim([500, 2500])
    xlabel("time (ms)",fontsize=15)
    ylabel("synaptic current",fontsize=15)

    if i == 4
      legend(loc=1,frameon=false,fontsize=15)
    end
end
tight_layout()
