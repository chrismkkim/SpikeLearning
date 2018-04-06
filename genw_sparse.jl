function genw_sparse(N,m,sd,p)

N    = Int(N);
w    = zeros((N,N));

# recurrent connections
for i=1:N
  for j=1:N
    d = Normal(m,sd)
    b = Bernoulli(p)

    w[i,j] = rand(d)*rand(b);
  end
end


# No autapse
for i=1:N
    w[i,i] = 0.;
end

return w

end
